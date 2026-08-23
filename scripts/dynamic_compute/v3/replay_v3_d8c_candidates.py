#!/usr/bin/env python3
"""Replay same-noise A1 L11/L13/L27 candidates for one D8C shard."""

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
from typing import Any, Mapping

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
    sha256_array,
    stream_sha256,
    validate_runtime_model_directory,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation import D8_TASK_IDS  # noqa: E402
from a1.vla.dynamic_compute.v3.fresh_confirmation_collection import (  # noqa: E402
    D8C_CANDIDATE_SCHEMA_VERSION,
    D8C_CONTEXT_SCHEMA_VERSION,
    D8C_REPLAY_LAYERS,
    D8C_ROLE,
    D8C_SUITE,
    load_fresh_task_calls,
    resolve_fresh_call_payload,
    validate_d8c_gpu_contract,
    validate_d8c_prerequisites,
)


RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d8c-candidate-shard-result.v1"
SHARD_COUNT = 4
REPLAY_SEED = 50_260_821


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--context-result", type=Path, required=True)
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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run(args: argparse.Namespace) -> None:
    prerequisites = validate_d8c_prerequisites(REPO_ROOT)
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D8C replay requires a clean worktree")
    if type(args.shard_index) is not int or not 0 <= args.shard_index < SHARD_COUNT:
        raise ValueError("D8C shard index must be in 0..3")
    expected_raw = (REPO_ROOT / "reports" / "v3_d8_fresh_raw").resolve()
    raw_root = args.raw_root.resolve(strict=True)
    if raw_root != expected_raw:
        raise PermissionError("D8C raw root path differs")
    expected_context = (
        REPO_ROOT / "reports" / "v3_d8_fresh_context" / "result.json"
    ).resolve()
    context_result_path = args.context_result.resolve(strict=True)
    if context_result_path != expected_context:
        raise PermissionError("D8C context result path differs")
    context_result = json.loads(context_result_path.read_text(encoding="utf-8"))
    if (
        context_result.get("status") != "PASS_V3_D8C_CONTEXT"
        or context_result.get("source_git_commit") != git_output("rev-parse", "HEAD")
        or context_result.get("source_worktree_dirty") is not False
    ):
        raise PermissionError("D8C context result has not passed on current commit")
    context_dir = context_result_path.parent
    context_payload_path = context_dir / str(context_result["payload"])
    if stream_sha256(context_payload_path) != context_result["payload_sha256"]:
        raise PermissionError("D8C context payload SHA-256 differs")
    context = torch.load(context_payload_path, map_location="cpu", weights_only=True)
    if (
        context.get("schema_version") != D8C_CONTEXT_SCHEMA_VERSION
        or context.get("role") != D8C_ROLE
        or context.get("suite") != D8C_SUITE
        or context.get("task_replicate_identity_is_runtime_input") is not False
        or context.get("layer27_is_runtime_input") is not False
    ):
        raise PermissionError("D8C context payload semantics differ")
    records_path = context_dir / str(context_result["records"])
    if stream_sha256(records_path) != context_result["records_sha256"]:
        raise PermissionError("D8C context records SHA-256 differs")
    context_records = _jsonl(records_path)
    context_by_index = {
        int(record["dataset_index"]): record for record in context_records
    }

    calls = []
    next_index = 0
    for task_id in D8_TASK_IDS:
        task_calls = load_fresh_task_calls(
            raw_root / f"task{task_id}",
            task_id=task_id,
            dataset_index_start=next_index,
        )
        calls.extend(task_calls)
        next_index += len(task_calls)
    assigned = [call for call in calls if call.dataset_index % SHARD_COUNT == args.shard_index]
    assigned_indices = [call.dataset_index for call in assigned]
    if not assigned or set(assigned_indices) - set(context_by_index):
        raise PermissionError("D8C shard assignment/context binding differs")

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_index):
        raise PermissionError("D8C CUDA_VISIBLE_DEVICES differs from assignment")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise PermissionError(f"D8C replay requires {name}=1")
    if not torch.cuda.is_available():
        raise RuntimeError("D8C candidate replay requires CUDA")
    torch.cuda.set_device(0)
    validate_d8c_gpu_contract(
        physical_gpu_index=args.physical_gpu_index,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        visible_gpu_count=torch.cuda.device_count(),
        expected_gpu_uuid=args.expected_gpu_uuid,
        observed_gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
    )
    checkpoint = args.checkpoint.resolve(strict=True)
    model_audit = validate_runtime_model_directory(checkpoint, args.model_attestation)
    expected_output = (
        REPO_ROOT / "reports" / "v3_d8_fresh_candidates" / f"shard{args.shard_index}"
    ).resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("D8C candidate shard output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("D8C refuses to overwrite candidate evidence")
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
    environment = {
        name: os.environ[name] for name in environment_names if name in os.environ
    }
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

    seed = REPLAY_SEED + args.shard_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    model = _load_frozen_a1(checkpoint, device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("D8C frozen A1 unexpectedly requires gradients")
    torch.cuda.reset_peak_memory_stats(device)

    actions: list[torch.Tensor] = []
    shared_inputs: list[torch.Tensor] = []
    records_path_out = incomplete / "candidate_records.jsonl"
    opened_payloads: list[Path] = []
    started = time.perf_counter()
    with records_path_out.open("w", encoding="utf-8") as record_file:
        for ordinal, call in enumerate(assigned, start=1):
            payload_path = resolve_fresh_call_payload(call)
            payload_sha256 = stream_sha256(payload_path)
            context_record = context_by_index[call.dataset_index]
            if (
                payload_sha256 != context_record["source_payload_sha256"]
                or call.task_id != int(context_record["task_id"])
                or call.replicate_id != int(context_record["replicate_id"])
                or call.policy_seed != int(context_record["policy_seed"])
                or call.cluster_key != str(context_record["cluster_key"])
                or call.call_ordinal != int(context_record["call_ordinal"])
            ):
                raise PermissionError("D8C context/source row binding differs")
            with np.load(payload_path, allow_pickle=False) as arrays:
                cpu_batch = replay_batch(arrays)
            opened_payloads.append(payload_path)
            shared_input = cpu_batch["teacher_exit_input_x"][0].float().contiguous()
            shared_hash = sha256_array(shared_input.numpy())
            batch = _move_batch(cpu_batch, device)
            replay_actions = []
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                for layer in D8C_REPLAY_LAYERS:
                    output_action = frozen_a1_action_forward(
                        model, None, batch, exit_layer=layer
                    ).normalized_action
                    replay_actions.append(output_action[0].float().cpu())
            candidate_actions = torch.stack(replay_actions).contiguous()
            after_hash = sha256_array(
                batch["teacher_exit_input_x"][0].float().cpu().numpy()
            )
            if after_hash != shared_hash:
                raise RuntimeError("D8C shared FM input mutated during replay")
            actions.append(candidate_actions)
            shared_inputs.append(shared_input)
            record = {
                "dataset_index": call.dataset_index,
                "task_id": call.task_id,
                "replicate_id": call.replicate_id,
                "policy_seed": call.policy_seed,
                "cluster_key": call.cluster_key,
                "call_ordinal": call.call_ordinal,
                "step_id": call.step_id,
                "source_payload_sha256": payload_sha256,
                "candidate_layers": list(D8C_REPLAY_LAYERS),
                "shared_fm_input_sha256": shared_hash,
                "shared_input_hash_stable": True,
                "candidate_actions_sha256": sha256_array(candidate_actions.numpy()),
                "layer27_role": "offline_same_noise_consistency_label_only",
                "router_scored": False,
                "active_control": False,
            }
            record_file.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
            record_file.flush()
            if ordinal == 1 or ordinal % 25 == 0 or ordinal == len(assigned):
                print(
                    f"D8C shard={args.shard_index} row={ordinal}/{len(assigned)} "
                    f"dataset_index={call.dataset_index}",
                    flush=True,
                )

    candidate_tensor = torch.stack(actions).float().contiguous()
    shared_tensor = torch.stack(shared_inputs).float().contiguous()
    if candidate_tensor.shape != (len(assigned), 3, 8, 7):
        raise RuntimeError("D8C candidate tensor geometry differs")
    if shared_tensor.shape != (len(assigned), 8, 7):
        raise RuntimeError("D8C shared input tensor geometry differs")
    payload = {
        "schema_version": D8C_CANDIDATE_SCHEMA_VERSION,
        "role": D8C_ROLE,
        "suite": D8C_SUITE,
        "D8_readiness_sha256": prerequisites["D8_readiness_sha256"],
        "shard_index": args.shard_index,
        "shard_count": SHARD_COUNT,
        "candidate_layers": torch.tensor(D8C_REPLAY_LAYERS, dtype=torch.long),
        "dataset_index": torch.tensor(assigned_indices, dtype=torch.long),
        "task_id": torch.tensor([call.task_id for call in assigned], dtype=torch.long),
        "replicate_id": torch.tensor(
            [call.replicate_id for call in assigned], dtype=torch.long
        ),
        "policy_seed": torch.tensor(
            [call.policy_seed for call in assigned], dtype=torch.long
        ),
        "cluster_keys": [call.cluster_key for call in assigned],
        "call_ordinal": torch.tensor(
            [call.call_ordinal for call in assigned], dtype=torch.long
        ),
        "shared_fm_input_x": shared_tensor,
        "candidate_actions": candidate_tensor,
        "layer27_is_consistency_teacher_only": True,
        "router_scored": False,
        "calibration_or_test_payload_accessed": False,
    }
    payload_path = incomplete / "fresh_candidates.pt"
    torch.save(payload, payload_path)
    checks = {
        "context_pass_and_source_binding_exact": True,
        "assigned_dataset_index_modulo_four_exact": all(
            index % SHARD_COUNT == args.shard_index for index in assigned_indices
        ),
        "only_fresh_generated_state_payloads_opened": len(opened_payloads)
        == len(assigned),
        "same_noise_l11_l13_l27_geometry_and_finiteness_exact": bool(
            torch.isfinite(candidate_tensor).all()
        ),
        "attested_frozen_a1_and_front_four_gpu": bool(model_audit),
        "official_episode_40_49_not_opened": all(
            ":episode" not in call.cluster_key for call in assigned
        ),
        "no_router_scoring_training_threshold_or_control": True,
    }
    passed = all(checks.values())
    result = {
        "status": "PASS_V3_D8C_CANDIDATE_SHARD"
        if passed
        else "FAIL_V3_D8C_CANDIDATE_SHARD",
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "role": D8C_ROLE,
        "suite": D8C_SUITE,
        "shard_index": args.shard_index,
        "rows": len(assigned),
        "candidate_layers": list(D8C_REPLAY_LAYERS),
        "context_result": str(context_result_path),
        "context_result_sha256": stream_sha256(context_result_path),
        "records": records_path_out.name,
        "records_sha256": stream_sha256(records_path_out),
        "payload": payload_path.name,
        "payload_sha256": stream_sha256(payload_path),
        "model_audit": model_audit,
        "physical_gpu_index": args.physical_gpu_index,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
        "checks": checks,
        "access_ledger": {
            "fresh_raw_call_payloads_opened": len(opened_payloads),
            "same_noise_candidate_rows": len(assigned),
            "official_episode_40_49_opened": False,
            "final_router_loaded": False,
            "confirmation_gate_inspected": False,
            "active_control": False,
        },
    }
    (incomplete / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("D8C candidate shard failed one or more gates")
    incomplete.rename(output)
    print("PASS_V3_D8C_CANDIDATE_SHARD", flush=True)


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
                        "status": "ABORT_V3_D8C_CANDIDATE_SHARD",
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
