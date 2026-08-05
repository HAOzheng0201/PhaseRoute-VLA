"""Release-time checks for the validated PhaseRoute-VLA runtime.

The learned route-then-solve routers remain research artifacts because their
sealed safety gates failed.  The release runtime is therefore the opt-in
RNG-preserving productive-exit plan (RP-PEP), whose frozen paired evaluation
is action- and trajectory-exact with the original A1 policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


RELEASE_SCHEMA_VERSION = "phase-route-vla.release-gate.v1"
RELEASE_METHOD = "rp_pep"
CHECKPOINT_SHA256 = (
    "dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"
)
CONFIG_SHA256 = (
    "9365d0a6ca6379a77877aaf46e170a7945f084c359560463edc14726965b04ca"
)
DATASET_STATISTICS_SHA256 = (
    "6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3"
)
THRESHOLD_SHA256 = (
    "5b3a0ee9f3851bf1b0c7f7e2b28bc61898ed0b4bd39f8752007719e9f26d7bd6"
)
PAIRED_RESULT_SHA256 = (
    "5e8c2fa2e50a30a5911b29bab796e50d624a2971da649b4aa82333ba9beefb16"
)
CHECKPOINT_RELATIVE_PATH = Path("model/libero_exit/model.pt")
CONFIG_RELATIVE_PATH = Path("model/libero_exit/config.yaml")
DATASET_STATISTICS_RELATIVE_PATH = Path(
    "model/libero_exit/dataset_statistics.json"
)
THRESHOLD_RELATIVE_PATH = Path(
    "model/libero_exit/exit_thresholds_libero_spatial_exp_1.0.json"
)
PAIRED_RESULT_RELATIVE_PATH = Path(
    "results/rp_pep_paired.json"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate_rp_pep_release(repo_root: str | Path) -> dict[str, Any]:
    """Validate immutable model, threshold, and paired-result evidence.

    This function performs no CUDA work and never enables the runtime.  A
    caller may only advertise the release as validated when every returned
    check is true.
    """

    root = Path(repo_root).resolve()
    checkpoint = root / CHECKPOINT_RELATIVE_PATH
    config = root / CONFIG_RELATIVE_PATH
    dataset_statistics = root / DATASET_STATISTICS_RELATIVE_PATH
    threshold = root / THRESHOLD_RELATIVE_PATH
    paired_result = root / PAIRED_RESULT_RELATIVE_PATH
    required_paths = (
        checkpoint,
        config,
        dataset_statistics,
        threshold,
        paired_result,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing release artifacts: {missing}")

    actual_hashes = {
        "checkpoint": sha256_file(checkpoint),
        "config": sha256_file(config),
        "dataset_statistics": sha256_file(dataset_statistics),
        "threshold": sha256_file(threshold),
        "paired_result": sha256_file(paired_result),
    }
    expected_hashes = {
        "checkpoint": CHECKPOINT_SHA256,
        "config": CONFIG_SHA256,
        "dataset_statistics": DATASET_STATISTICS_SHA256,
        "threshold": THRESHOLD_SHA256,
        "paired_result": PAIRED_RESULT_SHA256,
    }
    result = _load_json(paired_result)
    equivalence = result.get("equivalence", {})
    gates = result.get("gates", {})
    checks = {
        "checkpoint_sha256": actual_hashes["checkpoint"]
        == expected_hashes["checkpoint"],
        "config_sha256": actual_hashes["config"] == expected_hashes["config"],
        "dataset_statistics_sha256": actual_hashes["dataset_statistics"]
        == expected_hashes["dataset_statistics"],
        "threshold_sha256": actual_hashes["threshold"]
        == expected_hashes["threshold"],
        "paired_result_sha256": actual_hashes["paired_result"]
        == expected_hashes["paired_result"],
        "paired_result_status": result.get("status") == "PASS",
        "paired_result_scope": result.get("scope")
        == "m420b_rp_pep_paired_closed_loop_summary",
        "complete_pair_grid": int(result.get("paired_episodes", -1)) == 20
        and int(result.get("total_rollouts", -1)) == 40,
        "success_equivalence": int(equivalence.get("success_mismatches", -1)) == 0,
        "action_equivalence": int(
            equivalence.get("action_chunk_sha256_mismatches", -1)
        )
        == 0,
        "exit_equivalence": int(
            equivalence.get("exit_layer_sequence_mismatches", -1)
        )
        == 0,
        "policy_call_equivalence": int(
            equivalence.get("policy_call_count_mismatches", -1)
        )
        == 0,
        "all_frozen_gates": bool(gates) and all(bool(value) for value in gates.values()),
        "fm_reduction_at_least_35_percent": float(
            result.get("fm_solver_calls", {}).get("reduction_fraction", -1.0)
        )
        >= 0.35,
        "mean_latency_reduction_at_least_15_percent": float(
            result.get("policy_latency", {}).get(
                "weighted_mean_reduction_fraction", -1.0
            )
        )
        >= 0.15,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": status,
        "release_method": RELEASE_METHOD,
        "runtime_default_enabled": False,
        "learned_router_runtime_allowed": False,
        "repo_root": str(root),
        "artifacts": {
            "checkpoint": str(checkpoint),
            "config": str(config),
            "dataset_statistics": str(dataset_statistics),
            "threshold": str(threshold),
            "paired_result": str(paired_result),
        },
        "expected_sha256": expected_hashes,
        "actual_sha256": actual_hashes,
        "checks": checks,
        "frozen_metrics": {
            "paired_episodes": int(result.get("paired_episodes", 0)),
            "baseline_successes": int(result.get("baseline_successes", 0)),
            "rp_pep_successes": int(result.get("rp_pep_successes", 0)),
            "fm_reduction_fraction": float(
                result.get("fm_solver_calls", {}).get("reduction_fraction", 0.0)
            ),
            "mean_latency_reduction_fraction": float(
                result.get("policy_latency", {}).get(
                    "weighted_mean_reduction_fraction", 0.0
                )
            ),
            "median_latency_reduction_fraction": float(
                result.get("policy_latency", {}).get(
                    "median_reduction_fraction", 0.0
                )
            ),
        },
    }


__all__ = [
    "CHECKPOINT_SHA256",
    "CONFIG_SHA256",
    "DATASET_STATISTICS_SHA256",
    "PAIRED_RESULT_SHA256",
    "RELEASE_METHOD",
    "RELEASE_SCHEMA_VERSION",
    "THRESHOLD_SHA256",
    "sha256_file",
    "validate_rp_pep_release",
]
