#!/usr/bin/env python3
"""Run the frozen D4A motion/tail adapter without shadow selection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    build_gripper_v2_features,
)
from a1.vla.dynamic_compute.v3.shadow_signal_adapter import (  # noqa: E402
    D4A_CHECKPOINT_SHA256,
    D4A_SCHEMA_VERSION,
    D4A_TAIL_ARTIFACT_SHA256,
    adapt_shadow_signals,
    authenticated_weights_only_load,
    load_frozen_legacy_signal_state,
    stream_sha256,
    validate_v3_dataset_header,
)


OUTPUT = Path("reports/v3_d4a_signal_adapter")
D4A_VALIDATION = Path(
    "results/v3/v3_d4a_signal_adapter_contract_validation.json"
)
D4A_VALIDATION_SHA256 = (
    "e60cf377a86251fc5565c93fd6ff6d81df612392eec36c0ad75fa602f9ad244a"
)
D3_DATASET = Path(
    "reports/v3_d3_calibration_dataset/calibration_gripper_v2_dataset.pt"
)
D3_DATASET_SHA256 = (
    "5780e5949bc5b1ded15483ef84a08994ed099cd1a7604fb5c1c2082d7db4f005"
)
D3_CONTEXT = Path("reports/v3_d3_calibration_context/calibration_context.pt")
D3_CONTEXT_SHA256 = (
    "56edd68f73b3e9fbc0609a22a25a003dfa6ed02ca8ad2a313bf6852a4e81c506"
)
D3_SHARD_SHA256 = (
    "87f5a8b2a6eec1a2fe9ac49369ca5162a8c59555d5670bfb13f32b027e764eee",
    "f2710e023e4bce718743c1e802f6d4522adefe018133d789f587e096d3006a3c",
    "793f9de24121a10fd2fbe22646182e1675b00a1c1e1afc07725146a86da24bb7",
    "5fbfe3ff2cf6af76eac69381d585a06e6a701f4761852455a78c170047703111",
)
LEGACY_ROOT = Path("/data3/haozheng/A1/source")
LEGACY_CHECKPOINT = LEGACY_ROOT / (
    "reports/phase_route_v2_stage_c355_development_predictor_training_"
    "20260818_v1/development_checkpoint_candidates.pt"
)
LEGACY_TAIL = LEGACY_ROOT / (
    "reports/phase_route_v2_stage_c357_tail_calibration_finalizer_"
    "20260819_v1/tail_calibration_artifact.pt"
)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D4A adapter requires a clean worktree")
    validation_path = REPO_ROOT / D4A_VALIDATION
    if stream_sha256(validation_path) != D4A_VALIDATION_SHA256:
        raise PermissionError("V3-D4A contract validation SHA differs")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("status")
        != "PASS_V3_D4A_SIGNAL_ADAPTER_CONTRACT_FROZEN"
        or validation.get("next_stage", {}).get("formal_shadow_authorized")
        is not False
    ):
        raise PermissionError("V3-D4A contract validation boundary differs")
    dataset = authenticated_weights_only_load(
        REPO_ROOT / D3_DATASET,
        expected_sha256=D3_DATASET_SHA256,
        context="D3 dataset",
    )
    rows = validate_v3_dataset_header(dataset)
    context = authenticated_weights_only_load(
        REPO_ROOT / D3_CONTEXT,
        expected_sha256=D3_CONTEXT_SHA256,
        context="D3 context",
    )
    payloads = []
    for shard, expected in enumerate(D3_SHARD_SHA256):
        path = REPO_ROOT / (
            f"reports/v3_d3_calibration_candidates/shard{shard}/"
            "calibration_candidates.pt"
        )
        payloads.append(
            authenticated_weights_only_load(
                path, expected_sha256=expected, context=f"D3 candidate shard {shard}"
            )
        )
    merged = {
        name: torch.cat([payload[name] for payload in payloads]).contiguous()
        for name in (
            "dataset_index",
            "task_id",
            "episode_index",
            "call_ordinal",
            "candidate_actions",
        )
    }
    order = torch.argsort(merged["dataset_index"])
    merged = {name: value[order].contiguous() for name, value in merged.items()}
    for name in ("dataset_index", "task_id", "episode_index", "call_ordinal"):
        if not torch.equal(merged[name], context[name]):
            raise PermissionError(f"V3-D4A D3 row identity differs: {name}")
    rebuilt = build_gripper_v2_features(
        context["runtime_inputs"], merged["candidate_actions"][:, :2]
    ).reshape(rows, 97).contiguous()
    if not torch.equal(rebuilt, dataset["features"]):
        raise PermissionError("V3-D4A 97D feature rebuild/prefix differs")
    state = load_frozen_legacy_signal_state(LEGACY_CHECKPOINT, LEGACY_TAIL)
    started = time.perf_counter()
    signals = adapt_shadow_signals(
        state, dataset["features"], dataset["candidate_layer"]
    )
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D4A refuses to overwrite adapter evidence")
    incomplete.mkdir(parents=True)
    payload_path = incomplete / "adapter_signals.pt"
    payload = {
        "schema_version": D4A_SCHEMA_VERSION,
        "role": "calibration_v2",
        "source_row": dataset["source_row"].clone(),
        "task_id": dataset["task_id"].clone(),
        "episode_index": dataset["episode_index"].clone(),
        "candidate_layer": dataset["candidate_layer"].clone(),
        "motion_prediction": signals.motion_prediction.clone(),
        "tail_q90": signals.tail_q90.clone(),
        "tail_upper": signals.tail_upper.clone(),
        "motion_safe": signals.motion_safe.clone(),
        "tail_ucb_safe": signals.tail_ucb_safe.clone(),
        "checkpoint_sha256": D4A_CHECKPOINT_SHA256,
        "tail_artifact_sha256": D4A_TAIL_ARTIFACT_SHA256,
        "feature_prefix_exact": True,
        "threshold_search_or_fit": False,
        "shadow_decision_run": False,
        "active_control": False,
        "independent_test_payload_opened": False,
    }
    torch.save(payload, payload_path)
    payload_sha = stream_sha256(payload_path)
    result = {
        "status": "PASS_V3_D4A_SIGNAL_ADAPTER_ATTESTATION",
        "schema_version": "phase-route-vla.v3.d4a-adapter-result.v1",
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "role": "calibration_v2",
        "rows": rows,
        "source_calls": rows // 2,
        "decision_layers": [11, 13],
        "input_sha256": {
            "d3_dataset": D3_DATASET_SHA256,
            "d3_context": D3_CONTEXT_SHA256,
            "d3_candidate_shards": list(D3_SHARD_SHA256),
            "legacy_checkpoint": D4A_CHECKPOINT_SHA256,
            "legacy_tail_artifact": D4A_TAIL_ARTIFACT_SHA256,
        },
        "payload": "adapter_signals.pt",
        "payload_sha256": payload_sha,
        "elapsed_seconds": time.perf_counter() - started,
        "checks": {
            "d4a_contract_validation_current": True,
            "all_artifacts_authenticated_before_weights_only_load": True,
            "context_candidate_dataset_row_identity_exact": True,
            "v3_97d_features_rebuilt_exactly": True,
            "legacy_82d_slice_exact_prefix": True,
            "frozen_checkpoint_schema_and_claim_boundary_exact": True,
            "motion_and_tail_predictions_finite_positive": True,
            "thresholds_pre_frozen_without_search": True,
            "no_shadow_selection_counts_or_distribution_reported": True,
            "no_fit_gpu_independent_test_or_control": True,
        },
        "access_ledger": {
            "calibration_v2_tensor_payloads_opened": 6,
            "legacy_tensor_payloads_opened": 2,
            "model_fit_or_optimizer_step": 0,
            "runtime_threshold_search": 0,
            "shadow_decision_run": False,
            "independent_test_payload_opened": False,
            "gpu_query_or_initialization": 0,
            "active_control": False,
        },
        "next_stage": {
            "authorized": "V3-D4B_FORMAL_CALIBRATION_SHADOW_ONLY",
            "fresh_rollout_authorized": False,
            "active_control_authorized": False,
            "independent_test_authorized": False,
        },
        "claim_boundary": {
            "adapter_pass_is_shadow_result": False,
            "success_or_efficiency_improvement": False,
            "deployment_authorized": False,
            "superiority_claim_authorized": False,
        },
    }
    result_path = incomplete / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (incomplete / "result.sha256").write_text(
        f"{stream_sha256(result_path)}  result.json\n", encoding="utf-8"
    )
    shutil.move(str(incomplete), str(output))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
