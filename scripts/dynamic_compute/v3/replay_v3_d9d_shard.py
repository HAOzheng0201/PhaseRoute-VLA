#!/usr/bin/env python3
"""Replay one V3-D9D L11/L13/L27 same-noise truth shard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from typing import Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.config import TrainConfig  # noqa: E402
from a1.util import resource_path  # noqa: E402
from a1.vla.affordvla_early_exit import AffordVLAEarlyExit  # noqa: E402
from a1.vla.dynamic_compute.frozen_a1_distillation import (  # noqa: E402
    freeze_a1_for_action_distillation,
    frozen_a1_action_forward,
)
from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    D2_FM_STEPS,
    replay_batch,
    validate_runtime_model_directory,
)
from a1.vla.dynamic_compute.v3.paired_active_collection import (  # noqa: E402
    sha256_array,
    sha256_file,
)
from a1.vla.dynamic_compute.v3.same_noise_replay import (  # noqa: E402
    D9C_COLLECTION_SHA256,
    D9D_ACTION_THRESHOLD,
    D9D_EXPECTED_ROWS,
    D9D_OUTPUT_RELATIVE_PATH,
    D9D_REPLAY_LAYERS,
    D9D_SELECTED_REPLAY_ATOL,
    D9D_SEVERE_RATIO,
    D9D_SHARD_COUNT,
    D9D_SHARD_RESULT_SCHEMA_VERSION,
    D9D_SHARD_SCHEMA_VERSION,
    D9D_SHARD_STATUS,
    build_call_truth,
    hash_online_action,
    load_d9d_calls,
    validate_d9c_collection,
    validate_d9d_runner_readiness,
    validate_gpu_contract,
)


REPLAY_SEED = 80_260_821


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "model" / "v3_d2" / "libero_exit",
    )
    parser.add_argument("--model-attestation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _load_frozen_a1(checkpoint: Path, device: torch.device) -> AffordVLAEarlyExit:
    config = TrainConfig.load(resource_path(checkpoint, "config.yaml"), validate_paths=False)
    data_dir = Path(os.environ.get("DATA_DIR", str(REPO_ROOT)))
    config.model.vit_load_path = str(data_dir / "pretrained_image_encoders/vit-l-14-336.pt")
    config.model.llm_load_path = str(data_dir / "pretrained_llms/qwen2-7b.pt")
    config.model.tokenizer.tokenizer_dir = os.environ.get("HF_HOME", "")
    config.model.num_diffusion_inference_steps = D2_FM_STEPS
    config.model.init_device = str(device)
    model = AffordVLAEarlyExit(config.model)
    state = torch.load(
        resource_path(checkpoint, "model.pt"), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    return freeze_a1_for_action_distillation(model)


def _move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device=device, non_blocking=True) for name, value in batch.items()
    }


def _run(args: argparse.Namespace) -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D9D replay requires a clean frozen-runner worktree")
    collection = validate_d9c_collection(REPO_ROOT)
    readiness = validate_d9d_runner_readiness(REPO_ROOT)
    if collection["sha256"] != D9C_COLLECTION_SHA256:
        raise PermissionError("D9D collection binding differs")
    if type(args.shard_index) is not int or args.shard_index not in range(D9D_SHARD_COUNT):
        raise ValueError("D9D shard index must be in 0..3")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_index):
        raise PermissionError("D9D CUDA_VISIBLE_DEVICES differs from assignment")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise PermissionError(f"D9D replay requires {name}=1")
    if not torch.cuda.is_available():
        raise RuntimeError("D9D same-noise replay requires CUDA")
    torch.cuda.set_device(0)
    validate_gpu_contract(
        shard_index=args.shard_index,
        physical_gpu_index=args.physical_gpu_index,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        visible_gpu_count=torch.cuda.device_count(),
        expected_gpu_uuid=args.expected_gpu_uuid,
        observed_gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
    )
    checkpoint = args.checkpoint.resolve(strict=True)
    model_audit = validate_runtime_model_directory(checkpoint, args.model_attestation)
    expected_output = (
        REPO_ROOT / D9D_OUTPUT_RELATIVE_PATH / f"shard{args.shard_index}"
    ).resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("D9D shard output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("D9D refuses to overwrite replay evidence")
    incomplete.mkdir(parents=True, exist_ok=False)

    environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "DATA_DIR",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "VLA_CONFIG_YAML",
        "CUBLAS_WORKSPACE_CONFIG",
    )
    environment = {name: os.environ[name] for name in environment_names if name in os.environ}
    command = shlex.join(
        [
            "env",
            *(f"{key}={value}" for key, value in environment.items()),
            sys.executable,
            *sys.argv,
        ]
    )
    (incomplete / "command.txt").write_text(
        "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + command + "\n",
        encoding="utf-8",
    )

    calls = load_d9d_calls(REPO_ROOT)
    assigned = tuple(
        call for call in calls if call.dataset_index % D9D_SHARD_COUNT == args.shard_index
    )
    if (
        len(calls) != D9D_EXPECTED_ROWS
        or not assigned
        or any(call.dataset_index % D9D_SHARD_COUNT != args.shard_index for call in assigned)
    ):
        raise PermissionError("D9D modulo-four shard assignment differs")

    seed = REPLAY_SEED + args.shard_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    model = _load_frozen_a1(checkpoint, device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("D9D frozen A1 unexpectedly requires gradients")
    torch.cuda.reset_peak_memory_stats(device)

    candidate_rows: list[torch.Tensor] = []
    shared_rows: list[torch.Tensor] = []
    online_rows: list[torch.Tensor] = []
    distance_rows: list[float] = []
    full_unsafe_rows: list[bool] = []
    gripper_unsafe_rows: list[bool] = []
    severe_rows: list[bool] = []
    replay_error_rows: list[float] = []
    bit_exact_rows: list[bool] = []
    source_hashes: list[str] = []
    shared_hashes: list[str] = []
    candidate_hashes: list[str] = []
    online_hashes: list[str] = []
    records_path = incomplete / "truth_records.jsonl"
    started = time.perf_counter()
    with records_path.open("w", encoding="utf-8") as record_file:
        for ordinal, call in enumerate(assigned, start=1):
            if call.array_path.stat().st_size != call.array_bytes:
                raise PermissionError("D9D cache NPZ byte size changed")
            observed_source_sha = sha256_file(call.array_path)
            if observed_source_sha != call.array_sha256:
                raise PermissionError("D9D cache NPZ SHA-256 differs from inventory")
            with np.load(call.array_path, allow_pickle=False) as arrays:
                online_array = np.asarray(
                    arrays["teacher_normalized_action"], dtype=np.float32
                ).copy()
                trace_array = np.asarray(
                    arrays["teacher_exit_trace_action"], dtype=np.float32
                ).copy()
                if (
                    online_array.shape != (8, 7)
                    or trace_array.shape != (8, 7)
                    or not np.array_equal(online_array, trace_array)
                ):
                    raise PermissionError(
                        "D9D cached online action/selected trace binding differs"
                    )
                cpu_batch = replay_batch(arrays)
            shared_input = cpu_batch["teacher_exit_input_x"][0].float().contiguous()
            online_action = torch.from_numpy(online_array).float().contiguous()
            if shared_input.shape != (8, 7) or online_action.shape != (8, 7):
                raise RuntimeError("D9D cache action geometry differs")
            shared_hash = sha256_array(shared_input.numpy())
            online_hash = hash_online_action(online_action)
            batch = _move_batch(cpu_batch, device)
            replay_actions: list[torch.Tensor] = []
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                for layer in D9D_REPLAY_LAYERS:
                    action = frozen_a1_action_forward(
                        model, None, batch, exit_layer=layer
                    ).normalized_action
                    replay_actions.append(action[0].float().cpu())
            candidates = torch.stack(replay_actions).float().contiguous()
            after_hash = sha256_array(
                batch["teacher_exit_input_x"][0].float().cpu().numpy()
            )
            if after_hash != shared_hash:
                raise RuntimeError("D9D shared flow-matching input mutated during replay")
            truth = build_call_truth(
                candidates,
                selected_layer=call.selected_layer,
                online_selected_action=online_action,
            )
            if not truth.selected_replay_within_atol:
                raise RuntimeError(
                    "D9D selected-layer replay differs from the online selected action"
                )
            candidate_hash = sha256_array(candidates.numpy())
            candidate_rows.append(candidates)
            shared_rows.append(shared_input)
            online_rows.append(online_action)
            distance_rows.append(truth.full_action_distance)
            full_unsafe_rows.append(truth.full_action_unsafe)
            gripper_unsafe_rows.append(truth.gripper_unsafe)
            severe_rows.append(truth.severe_full_action)
            replay_error_rows.append(truth.selected_replay_max_abs_error)
            bit_exact_rows.append(truth.selected_replay_bit_exact)
            source_hashes.append(observed_source_sha)
            shared_hashes.append(shared_hash)
            candidate_hashes.append(candidate_hash)
            online_hashes.append(online_hash)
            record = {
                "dataset_index": call.dataset_index,
                "task_id": call.task_id,
                "episode_index": call.episode_index,
                "seed": call.seed,
                "canonical_key": call.canonical_key,
                "call_ordinal": call.call_ordinal,
                "step_id": call.step_id,
                "online_selected_layer": call.selected_layer,
                "source_npz_path": call.array_path.relative_to(REPO_ROOT).as_posix(),
                "source_npz_bytes": call.array_bytes,
                "source_npz_sha256": observed_source_sha,
                "shared_fm_input_sha256": shared_hash,
                "online_selected_action_sha256": online_hash,
                "candidate_layers": list(D9D_REPLAY_LAYERS),
                "candidate_actions_sha256": candidate_hash,
                "selected_candidate_index": truth.selected_candidate_index,
                "selected_replay_max_abs_error": truth.selected_replay_max_abs_error,
                "selected_replay_atol": D9D_SELECTED_REPLAY_ATOL,
                "selected_replay_within_atol": truth.selected_replay_within_atol,
                "selected_replay_bit_exact": truth.selected_replay_bit_exact,
                "full_action_distance_selected_vs_L27": truth.full_action_distance,
                "full_action_unsafe_threshold": D9D_ACTION_THRESHOLD,
                "full_action_unsafe": truth.full_action_unsafe,
                "gripper_XOR_selected_vs_L27": truth.gripper_unsafe,
                "severe_full_action_ratio": D9D_SEVERE_RATIO,
                "severe_full_action": truth.severe_full_action,
                "layer27_role": "offline_same_noise_consistency_teacher_only_not_expert",
                "router_scored": False,
                "LIBERO_environment_created": False,
                "environment_action_executed": False,
                "online_selected_action_modified": False,
            }
            record_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            record_file.flush()
            if ordinal == 1 or ordinal % 25 == 0 or ordinal == len(assigned):
                print(
                    f"D9D shard={args.shard_index} row={ordinal}/{len(assigned)} "
                    f"dataset_index={call.dataset_index}",
                    flush=True,
                )

    candidate_tensor = torch.stack(candidate_rows).float().contiguous()
    shared_tensor = torch.stack(shared_rows).float().contiguous()
    online_tensor = torch.stack(online_rows).float().contiguous()
    rows = len(assigned)
    if (
        candidate_tensor.shape != (rows, 3, 8, 7)
        or shared_tensor.shape != (rows, 8, 7)
        or online_tensor.shape != (rows, 8, 7)
    ):
        raise RuntimeError("D9D replay tensor geometry differs")
    payload = {
        "schema_version": D9D_SHARD_SCHEMA_VERSION,
        "role": "D9D_same_noise_truth_only",
        "suite": "libero_10",
        "D9C_collection_sha256": collection["sha256"],
        "D9D_runner_readiness_sha256": readiness["sha256"],
        "shard_index": args.shard_index,
        "shard_count": D9D_SHARD_COUNT,
        "candidate_layers": torch.tensor(D9D_REPLAY_LAYERS, dtype=torch.long),
        "dataset_index": torch.tensor([call.dataset_index for call in assigned]),
        "task_id": torch.tensor([call.task_id for call in assigned]),
        "episode_index": torch.tensor([call.episode_index for call in assigned]),
        "seed": torch.tensor([call.seed for call in assigned]),
        "canonical_keys": [call.canonical_key for call in assigned],
        "call_ordinal": torch.tensor([call.call_ordinal for call in assigned]),
        "step_id": torch.tensor([call.step_id for call in assigned]),
        "selected_layer": torch.tensor([call.selected_layer for call in assigned]),
        "candidate_actions": candidate_tensor,
        "shared_fm_input_x": shared_tensor,
        "online_selected_action": online_tensor,
        "full_action_distance": torch.tensor(distance_rows, dtype=torch.float64),
        "full_action_unsafe": torch.tensor(full_unsafe_rows, dtype=torch.bool),
        "gripper_unsafe": torch.tensor(gripper_unsafe_rows, dtype=torch.bool),
        "severe_full_action": torch.tensor(severe_rows, dtype=torch.bool),
        "selected_replay_max_abs_error": torch.tensor(
            replay_error_rows, dtype=torch.float64
        ),
        "selected_replay_bit_exact": torch.tensor(bit_exact_rows, dtype=torch.bool),
        "source_npz_sha256": source_hashes,
        "shared_fm_input_sha256": shared_hashes,
        "candidate_actions_sha256": candidate_hashes,
        "online_selected_action_sha256": online_hashes,
        "full_action_threshold": D9D_ACTION_THRESHOLD,
        "severe_ratio": D9D_SEVERE_RATIO,
        "layer27_is_consistency_teacher_only": True,
        "router_scored": False,
        "active_control": False,
        "D9_gate_evaluated": False,
    }
    payload_path = incomplete / "truth_payload.pt"
    torch.save(payload, payload_path)
    checks = {
        "D9C_collection_binding_exact": bool(collection),
        "D9D_runner_readiness_exact": bool(readiness),
        "all_source_NPZ_inventory_hashes_exact": len(source_hashes) == rows,
        "modulo_four_assignment_exact": all(
            call.dataset_index % D9D_SHARD_COUNT == args.shard_index for call in assigned
        ),
        "same_noise_L11_L13_L27_geometry_and_finiteness_exact": bool(
            torch.isfinite(candidate_tensor).all()
        ),
        "shared_FM_input_hash_stable": len(shared_hashes) == rows,
        "online_selected_action_replay_within_frozen_atol": all(
            value <= D9D_SELECTED_REPLAY_ATOL for value in replay_error_rows
        ),
        "attested_frozen_A1_and_front_four_GPU": bool(model_audit),
        "all_policy_calls_replayed_not_only_early_calls": rows
        == sum(1 for call in calls if call.dataset_index % D9D_SHARD_COUNT == args.shard_index),
        "no_environment_router_action_mutation_or_gate_aggregate": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"D9D shard checks failed: {checks}")
    result = {
        "status": D9D_SHARD_STATUS,
        "schema_version": D9D_SHARD_RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "shard_index": args.shard_index,
        "shard_count": D9D_SHARD_COUNT,
        "rows": rows,
        "candidate_layers": list(D9D_REPLAY_LAYERS),
        "D9C_collection": {
            key: value for key, value in collection.items() if key != "arm_payload_binding"
        },
        "D9D_runner_readiness": readiness,
        "records": records_path.name,
        "records_sha256": sha256_file(records_path),
        "payload": payload_path.name,
        "payload_sha256": sha256_file(payload_path),
        "model_audit": model_audit,
        "physical_gpu_index": args.physical_gpu_index,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
        "checks": checks,
        "access_ledger": {
            "cache_NPZ_payloads_opened": rows,
            "candidate_replays": rows * len(D9D_REPLAY_LAYERS),
            "LIBERO_environments_created": 0,
            "environment_actions_executed": 0,
            "routers_loaded": 0,
            "D9_gate_aggregate_calls": 0,
        },
        "claim_boundary": {
            "per_call_truth_created": True,
            "success_safety_efficiency_aggregate_computed": False,
            "D9_primary_gate_evaluated": False,
        },
    }
    (incomplete / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    incomplete.rename(output)
    print(D9D_SHARD_STATUS, flush=True)


def main() -> None:
    args = parse_args()
    try:
        _run(args)
    except BaseException as error:
        incomplete = args.output_dir.resolve().with_name(args.output_dir.name + ".incomplete")
        if incomplete.is_dir() and not (incomplete / "abort.json").exists():
            (incomplete / "abort.json").write_text(
                json.dumps(
                    {
                        "status": "ABORT_V3_D9D_SAME_NOISE_TRUTH_SHARD",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
