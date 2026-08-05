"""Fail-closed summary for the M4.23 three-policy fixed-observation profiles."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_compute.profile_m423_fixed_observations import (
    POLICIES,
    summarize,
)
from scripts.dynamic_compute.replay_m420b_rp_pep import normalize_gpu_uuid


EXPECTED_ORDERS = {
    0: ("early_exit", "rp_pep", "full_depth"),
    1: ("rp_pep", "full_depth", "early_exit"),
    2: ("full_depth", "early_exit", "rp_pep"),
    3: ("full_depth", "rp_pep", "early_exit"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-result", type=Path, action="append", required=True)
    parser.add_argument("--m422-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def timed_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["task_id"]),
        str(row["episode_id"]),
        int(row["step_id"]),
        int(row["teacher_exit_layer"]),
        int(row["repeat"]),
    )


def component_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["task_id"]),
        str(row["episode_id"]),
        int(row["step_id"]),
        int(row["teacher_exit_layer"]),
    )


def indexed(rows: Sequence[Mapping[str, Any]], key_fn) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    output: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = key_fn(row)
        if key in output:
            raise ValueError(f"duplicate profile row: {key}")
        output[key] = row
    return output


def paired_ratio(
    numerator: Mapping[tuple[Any, ...], Mapping[str, Any]],
    denominator: Mapping[tuple[Any, ...], Mapping[str, Any]],
    field: str,
) -> list[float]:
    if numerator.keys() != denominator.keys():
        raise ValueError("paired profile grids differ")
    ratios = []
    for key in sorted(numerator):
        denominator_value = float(denominator[key][field])
        if denominator_value <= 0:
            raise ValueError(f"nonpositive denominator for {key}: {denominator_value}")
        ratios.append(float(numerator[key][field]) / denominator_value)
    return ratios


def _load_m422_task_effect(result: Mapping[str, Any]) -> dict[str, Any]:
    successes = result.get("successes")
    if not isinstance(successes, Mapping):
        raise ValueError("M4.22 result is missing successes")
    rp = int(successes["rp_pep"])
    full = int(successes["full_depth"])
    records_value = result.get(
        "records", result.get("paired_states", result.get("analyzed_rollouts"))
    )
    if records_value is None:
        raise ValueError("M4.22 result is missing its paired-state count")
    records = int(records_value)
    return {
        "source_scope": result.get("scope"),
        "records": records,
        "rp_pep_successes": rp,
        "full_depth_successes": full,
        "rp_pep_not_lower_observed": rp >= full,
    }


def summarize_profiles(
    profile_items: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    m422_result: Mapping[str, Any],
) -> dict[str, Any]:
    sessions: dict[tuple[int, str], tuple[str, Mapping[str, Any]]] = {}
    for source, result in profile_items:
        if result.get("scope") != "m423_fixed_observation_policy_profile":
            raise ValueError(f"unexpected profile scope: {source}")
        if result.get("status") != "PASS":
            raise ValueError(f"non-PASS policy profile: {source}")
        gpu = int(result["physical_gpu_index"])
        policy = str(result["policy"])
        if gpu not in EXPECTED_ORDERS or policy not in POLICIES:
            raise ValueError(f"unexpected GPU/policy session: {gpu}/{policy}")
        key = (gpu, policy)
        if key in sessions:
            raise ValueError(f"duplicate GPU/policy session: {key}")
        sessions[key] = (source, result)

    expected_keys = {(gpu, policy) for gpu in EXPECTED_ORDERS for policy in POLICIES}
    if sessions.keys() != expected_keys:
        missing = sorted(expected_keys - sessions.keys())
        extra = sorted(sessions.keys() - expected_keys)
        raise ValueError(f"profile session grid differs; missing={missing}, extra={extra}")

    selection_hashes = {result["selection_sha256"] for _, result in sessions.values()}
    checkpoint_hashes = {result["checkpoint_sha256"] for _, result in sessions.values()}
    if len(selection_hashes) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("selection/checkpoint hashes differ across sessions")

    order_mismatches = []
    uuid_mismatches = []
    visible_uuids: dict[int, str] = {}
    local_count_mismatches = []
    for (gpu, policy), (source, result) in sessions.items():
        expected_position = EXPECTED_ORDERS[gpu].index(policy) + 1
        if int(result["order_position"]) != expected_position:
            order_mismatches.append(
                {"source": source, "expected": expected_position, "actual": result["order_position"]}
            )
        host_uuid = normalize_gpu_uuid(result["physical_gpu_uuid_nvidia_smi"])
        visible_uuid = normalize_gpu_uuid(result["physical_gpu_uuid_visible"])
        if host_uuid != visible_uuid:
            uuid_mismatches.append(source)
        previous = visible_uuids.setdefault(gpu, visible_uuid)
        if previous != visible_uuid:
            uuid_mismatches.append(source)
        if len(result["timed_samples"]) != 24 or len(result["component_samples"]) != 12:
            local_count_mismatches.append(source)
    unique_gpu_uuids = set(visible_uuids.values())

    action_mismatches = []
    exit_mismatches = []
    per_gpu: dict[str, Any] = {}
    all_policy_timed: dict[str, list[Mapping[str, Any]]] = {policy: [] for policy in POLICIES}
    all_policy_components: dict[str, list[Mapping[str, Any]]] = {policy: [] for policy in POLICIES}
    latency_all_ratios: dict[str, list[float]] = {
        "early_over_full": [],
        "rp_pep_over_full": [],
        "rp_pep_over_early": [],
    }
    by_actual_exit: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for gpu in sorted(EXPECTED_ORDERS):
        timed = {
            policy: indexed(sessions[(gpu, policy)][1]["timed_samples"], timed_key)
            for policy in POLICIES
        }
        components = {
            policy: indexed(
                sessions[(gpu, policy)][1]["component_samples"], component_key
            )
            for policy in POLICIES
        }
        if not (timed["early_exit"].keys() == timed["rp_pep"].keys() == timed["full_depth"].keys()):
            raise ValueError(f"timed grids differ on GPU{gpu}")
        if not (
            components["early_exit"].keys()
            == components["rp_pep"].keys()
            == components["full_depth"].keys()
        ):
            raise ValueError(f"component grids differ on GPU{gpu}")
        for key in sorted(timed["early_exit"]):
            early = timed["early_exit"][key]
            rp_pep = timed["rp_pep"][key]
            if early["action_sha256"] != rp_pep["action_sha256"]:
                action_mismatches.append({"gpu": gpu, "key": list(key)})
            if int(early["exit_layer"]) != int(rp_pep["exit_layer"]):
                exit_mismatches.append({"gpu": gpu, "key": list(key)})
            actual_exit = int(early["exit_layer"])
            for policy in POLICIES:
                by_actual_exit[actual_exit][f"{policy}_latency_ms"].append(
                    float(timed[policy][key]["cuda_latency_ms"])
                )
                by_actual_exit[actual_exit][f"{policy}_fm_calls"].append(
                    float(timed[policy][key]["fm_calls"])
                )
            by_actual_exit[actual_exit]["rp_pep_over_full"].append(
                float(rp_pep["cuda_latency_ms"])
                / float(timed["full_depth"][key]["cuda_latency_ms"])
            )
            by_actual_exit[actual_exit]["rp_pep_over_early"].append(
                float(rp_pep["cuda_latency_ms"])
                / float(early["cuda_latency_ms"])
            )

        early_full = paired_ratio(timed["early_exit"], timed["full_depth"], "cuda_latency_ms")
        rp_full = paired_ratio(timed["rp_pep"], timed["full_depth"], "cuda_latency_ms")
        rp_early = paired_ratio(timed["rp_pep"], timed["early_exit"], "cuda_latency_ms")
        latency_all_ratios["early_over_full"].extend(early_full)
        latency_all_ratios["rp_pep_over_full"].extend(rp_full)
        latency_all_ratios["rp_pep_over_early"].extend(rp_early)
        memory = {
            policy: sessions[(gpu, policy)][1]["memory_bytes"]
            for policy in POLICIES
        }
        per_gpu[str(gpu)] = {
            "uuid": visible_uuids[gpu],
            "order": list(EXPECTED_ORDERS[gpu]),
            "paired_cuda_latency_ratio": {
                "early_over_full": summarize(early_full),
                "rp_pep_over_full": summarize(rp_full),
                "rp_pep_over_early": summarize(rp_early),
            },
            "timed_cuda_latency_ms": {
                policy: summarize(row["cuda_latency_ms"] for row in timed[policy].values())
                for policy in POLICIES
            },
            "memory_bytes": memory,
        }
        for policy in POLICIES:
            all_policy_timed[policy].extend(timed[policy].values())
            all_policy_components[policy].extend(components[policy].values())

    policy_summary: dict[str, Any] = {}
    for policy in POLICIES:
        timed_rows = all_policy_timed[policy]
        component_rows = all_policy_components[policy]
        policy_summary[policy] = {
            "timed_samples": len(timed_rows),
            "component_samples": len(component_rows),
            "cuda_latency_ms": summarize(row["cuda_latency_ms"] for row in timed_rows),
            "wall_latency_ms": summarize(row["wall_latency_ms"] for row in timed_rows),
            "fm_calls": {
                "total": int(sum(int(row["fm_calls"]) for row in timed_rows)),
                "per_call": summarize(row["fm_calls"] for row in timed_rows),
            },
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

    by_actual_exit_summary = {
        str(exit_layer): {
            "paired_samples": len(values["rp_pep_over_full"]),
            "cuda_latency_ms": {
                policy: summarize(values[f"{policy}_latency_ms"])
                for policy in POLICIES
            },
            "fm_calls": {
                policy: summarize(values[f"{policy}_fm_calls"])
                for policy in POLICIES
            },
            "paired_cuda_latency_ratio": {
                "rp_pep_over_full": summarize(values["rp_pep_over_full"]),
                "rp_pep_over_early": summarize(values["rp_pep_over_early"]),
            },
        }
        for exit_layer, values in sorted(by_actual_exit.items())
    }

    task_effect = _load_m422_task_effect(m422_result)
    equivalence = not action_mismatches and not exit_mismatches
    latency_not_higher_every_gpu = all(
        per_gpu[str(gpu)]["paired_cuda_latency_ratio"]["rp_pep_over_full"]["median"] <= 1.0
        for gpu in EXPECTED_ORDERS
    )
    memory_not_higher_every_gpu = all(
        int(sessions[(gpu, "rp_pep")][1]["memory_bytes"]["timed_peak_allocated"])
        <= int(sessions[(gpu, "full_depth")][1]["memory_bytes"]["timed_peak_allocated"])
        for gpu in EXPECTED_ORDERS
    )
    pareto_gates = {
        "frozen_closed_loop_task_effect_not_lower_than_full_depth": task_effect[
            "rp_pep_not_lower_observed"
        ],
        "rp_pep_median_latency_not_higher_on_every_gpu": latency_not_higher_every_gpu,
        "rp_pep_peak_allocated_memory_not_higher_on_every_gpu": memory_not_higher_every_gpu,
        "early_rp_pep_strict_output_equivalence": equivalence,
    }
    overall_pareto = all(pareto_gates.values())

    engineering_checks = {
        "complete_4gpu_3policy_grid": len(sessions) == 12,
        "all_local_profiles_passed": True,
        "same_selection_sha256": len(selection_hashes) == 1,
        "same_checkpoint_sha256": len(checkpoint_hashes) == 1,
        "preregistered_order": not order_mismatches,
        "physical_gpu_uuid_match": not uuid_mismatches,
        "four_unique_front_gpu_uuids": len(unique_gpu_uuids) == 4,
        "profile_sample_counts": not local_count_mismatches,
        "early_rp_pep_action_equivalence": not action_mismatches,
        "early_rp_pep_exit_equivalence": not exit_mismatches,
        "full_depth_single_fm_solve": all(
            bool(sessions[(gpu, "full_depth")][1]["local_checks"]["full_depth_single_solve"])
            for gpu in EXPECTED_ORDERS
        ),
        "component_events_consistent": all(
            bool(result["local_checks"]["component_events_consistent"])
            for _, result in sessions.values()
        ),
    }
    status = "PASS" if all(engineering_checks.values()) else "FAIL"
    return {
        "status": status,
        "scope": "m423_full_depth_pareto_challenge_summary",
        "sessions": len(sessions),
        "physical_gpus": sorted(EXPECTED_ORDERS),
        "unique_gpu_uuids": sorted(unique_gpu_uuids),
        "selection_sha256": next(iter(selection_hashes)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "engineering_checks": engineering_checks,
        "audit_counters": {
            "order_mismatches": len(order_mismatches),
            "uuid_mismatches": len(uuid_mismatches),
            "sample_count_mismatches": len(local_count_mismatches),
            "early_rp_pep_action_mismatches": len(action_mismatches),
            "early_rp_pep_exit_mismatches": len(exit_mismatches),
        },
        "policy_summary": policy_summary,
        "paired_cuda_latency_ratio": {
            name: summarize(values) for name, values in latency_all_ratios.items()
        },
        "by_actual_early_exit_layer": by_actual_exit_summary,
        "per_gpu": per_gpu,
        "task_effect_control": task_effect,
        "pareto_challenge": {
            "status": "PASS" if overall_pareto else "NOT_MET",
            "overall_pareto_improvement": overall_pareto,
            "gates": pareto_gates,
        },
        "diagnosis": {
            "rp_pep_extra_fm_solves_per_call_vs_full_depth": (
                policy_summary["rp_pep"]["fm_calls"]["per_call"]["mean"] - 1.0
            ),
            "rp_pep_fm_head_mean_ms_minus_full_depth": (
                policy_summary["rp_pep"]["component_ms"]["fm_head"]["mean"]
                - policy_summary["full_depth"]["component_ms"]["fm_head"]["mean"]
            ),
            "rp_pep_transformer_mean_ms_minus_full_depth": (
                policy_summary["rp_pep"]["component_ms"]["transformer"]["mean"]
                - policy_summary["full_depth"]["component_ms"]["transformer"]["mean"]
            ),
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    items = [(str(path.resolve()), json.loads(path.read_text())) for path in args.profile_result]
    m422 = json.loads(args.m422_result.read_text())
    result = summarize_profiles(items, m422_result=m422)
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
