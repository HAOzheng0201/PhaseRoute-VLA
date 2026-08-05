"""Collect causal layer-11/13 router features without running the FM head."""

from __future__ import annotations

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

from scripts.dynamic_compute.replay_m420b_rp_pep import (  # noqa: E402
    _load_call_inputs,
    _load_manifest_entries,
    normalize_gpu_uuid,
)


FEATURE_SCHEMA = "phase-route-vla.m425-causal-route-features.v1"
EXPECTED_CACHE_SCHEMA = "phase-route-vla.vision-teacher-call.v3"
FEATURE_LAYERS = (11, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--expected-task-id", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def canonical_identity(cache_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_dir": str(cache_dir.resolve()),
        "array_path": str(record["array_path"]),
        "episode_id": str(record["episode_id"]),
        "task_id": int(record["task_id"]),
        "step_id": int(record["step_id"]),
        "teacher_exit_layer": int(record["teacher_exit_layer"]),
    }


def identity_sha256(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(identity), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_descriptor(cache_dir: Path) -> dict[str, Any]:
    manifest = cache_dir / "manifest.jsonl"
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty teacher cache manifest: {manifest}")
    task_ids = {int(row["task_id"]) for row in rows}
    if len(task_ids) != 1:
        raise ValueError(f"cache manifest mixes tasks: {manifest}")
    return {
        "task_id": next(iter(task_ids)),
        "path": str(cache_dir.resolve()),
        "records": len(rows),
        "manifest_sha256": sha256_file(manifest),
    }


def descriptor_sha256(descriptors: list[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        sorted((dict(item) for item in descriptors), key=lambda item: int(item["task_id"])),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_proprio_hidden(hidden: torch.Tensor, proprio_token_idx: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError("hidden state must have shape [B, L, D]")
    indices = proprio_token_idx.to(device=hidden.device, dtype=torch.long).reshape(-1)
    if indices.shape[0] != hidden.shape[0]:
        raise ValueError("proprio_token_idx batch differs from hidden state")
    if bool(torch.any(indices < 0)) or bool(torch.any(indices >= hidden.shape[1])):
        raise ValueError("proprio token index is outside hidden sequence")
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    output = hidden[batch, indices]
    if output.ndim != 2 or not bool(torch.isfinite(output).all()):
        raise RuntimeError("collected router hidden is invalid")
    return output


def collect_one(
    model: torch.nn.Module,
    inputs: Mapping[str, Any],
    *,
    amp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_inputs = dict(inputs)
    # Cached-call helpers carry the baseline default explicitly.  Replace it
    # in the mapping so the feature collector has one unambiguous value.
    forward_inputs["exit_id"] = 13
    forward_inputs["output_hidden_states"] = True
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled
    ):
        output = model.forward(**forward_inputs)
    if int(output.exit_layer) != 13:
        raise RuntimeError("feature forward did not stop at layer13")
    if output.attn_key_values is None or len(output.attn_key_values) != 14:
        raise RuntimeError("feature forward returned an invalid KV depth")
    if output.hidden_states is None or len(output.hidden_states) != 14:
        raise RuntimeError("feature forward returned an invalid hidden-state trace")
    # hidden_states[k] is the input to block k, hence item 12 is post-block11.
    hidden11 = output.hidden_states[12]
    hidden13 = output.last_hidden_state
    if hidden13 is None:
        raise RuntimeError("feature forward omitted last_hidden_state")
    index = forward_inputs["proprio_token_idx"]
    return extract_proprio_hidden(hidden11, index), extract_proprio_hidden(hidden13, index)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.physical_gpu_index not in (0, 1, 2, 3):
        raise ValueError("M4.25 only permits physical GPUs 0-3")
    expected_tasks = tuple(sorted(set(args.expected_task_id)))
    if len(expected_tasks) != len(args.expected_task_id):
        raise ValueError("expected task IDs must be unique")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("M4.25 collection requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    visible_uuid = str(torch.cuda.get_device_properties(device).uuid)
    if normalize_gpu_uuid(visible_uuid) != normalize_gpu_uuid(args.expected_gpu_uuid):
        raise RuntimeError(
            f"GPU UUID mismatch: expected {args.expected_gpu_uuid}, visible {visible_uuid}"
        )

    cache_dirs = [path.resolve() for path in args.cache_dir]
    descriptors = [manifest_descriptor(path) for path in cache_dirs]
    actual_tasks = tuple(sorted(int(item["task_id"]) for item in descriptors))
    if actual_tasks != expected_tasks:
        raise ValueError(
            f"cache tasks differ from frozen shard: {actual_tasks} != {expected_tasks}"
        )
    entries = _load_manifest_entries(cache_dirs, args.checkpoint_sha256)
    if any(record.get("schema_version") != EXPECTED_CACHE_SCHEMA for _, record in entries):
        raise ValueError("M4.25 requires only v3 teacher caches")
    entries = sorted(
        entries,
        key=lambda item: (
            int(item[1]["task_id"]),
            str(item[1]["episode_id"]),
            int(item[1]["step_id"]),
            str(item[1]["array_path"]),
        ),
    )
    identities = [canonical_identity(cache_dir, record) for cache_dir, record in entries]
    identity_hashes = [identity_sha256(item) for item in identities]
    if len(set(identity_hashes)) != len(identity_hashes):
        raise ValueError("feature shard contains duplicate canonical identities")

    from robot_experiments.libero.eval_libero_early_exit import (
        GenerateConfig,
        initialize_and_load_model,
    )
    from robot_experiments.robot_utils import set_seed_everywhere

    set_seed_everywhere(args.seed)
    cfg = GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint.resolve()),
        task_suite_name="libero_spatial",
        action_head_flow_matching_inference_steps=10,
        exit_interval=2,
        steps_per_stage=1,
        threshold_type="cosine",
        exit_dist="exp",
        exit_ratio=1.0,
        use_wandb=False,
        save_rollout_video=False,
        seed=args.seed,
    )
    model, device, _ = initialize_and_load_model(cfg)
    model_dtype = model.transformer.wte.embedding.dtype
    amp_enabled = not args.disable_amp

    layer11_rows: list[np.ndarray] = []
    layer13_rows: list[np.ndarray] = []
    proprio_rows: list[np.ndarray] = []
    for index, (cache_dir, record) in enumerate(entries):
        inputs, _, _ = _load_call_inputs(
            cache_dir, record, device=device, model_dtype=model_dtype
        )
        hidden11, hidden13 = collect_one(model, inputs, amp_enabled=amp_enabled)
        layer11_rows.append(hidden11[0].float().cpu().numpy())
        layer13_rows.append(hidden13[0].float().cpu().numpy())
        proprio = inputs["action_proprio"].reshape(inputs["action_proprio"].shape[0], -1)
        proprio_rows.append(proprio[0].float().cpu().numpy())
        print(
            f"[{index + 1:03d}/{len(entries):03d}] task={record['task_id']} "
            f"step={record['step_id']} teacher_route={record['teacher_exit_layer']} "
            f"h11_norm={float(hidden11.float().norm()):.3f} "
            f"h13_norm={float(hidden13.float().norm()):.3f}",
            flush=True,
        )

    layer11 = np.stack(layer11_rows).astype(np.float16)
    layer13 = np.stack(layer13_rows).astype(np.float16)
    proprio = np.stack(proprio_rows).astype(np.float32)
    expected_shape = (len(entries), int(model.config.d_model))
    if layer11.shape != expected_shape or layer13.shape != expected_shape:
        raise RuntimeError(
            f"unexpected collected hidden shapes: {layer11.shape}, {layer13.shape}"
        )
    if proprio.shape[0] != len(entries) or not all(
        np.isfinite(value).all() for value in (layer11, layer13, proprio)
    ):
        raise RuntimeError("collected feature arrays are incomplete or non-finite")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = args.output_dir / "features.npz"
    np.savez_compressed(
        arrays_path,
        layer11_hidden=layer11,
        layer13_hidden=layer13,
        normalized_proprio=proprio,
        task_id=np.asarray([item["task_id"] for item in identities], dtype=np.int16),
        step_id=np.asarray([item["step_id"] for item in identities], dtype=np.int32),
        teacher_route=np.asarray(
            [item["teacher_exit_layer"] for item in identities], dtype=np.int16
        ),
        identity_sha256=np.asarray(identity_hashes, dtype="S64"),
    )
    arrays_sha = sha256_file(arrays_path)
    status = "PASS"
    local_checks = {
        "record_count": len(entries) == sum(int(item["records"]) for item in descriptors),
        "unique_identity": len(set(identity_hashes)) == len(identity_hashes),
        "layer11_shape": layer11.shape == expected_shape,
        "layer13_shape": layer13.shape == expected_shape,
        "finite_features": all(np.isfinite(value).all() for value in (layer11, layer13)),
        "finite_proprio": bool(np.isfinite(proprio).all()),
        "expected_tasks": actual_tasks == expected_tasks,
    }
    if not all(local_checks.values()):
        status = "FAIL"
    result = {
        "status": status,
        "scope": "m425_causal_route_feature_shard",
        "schema_version": FEATURE_SCHEMA,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(git_output("status", "--porcelain")),
        "source_status_sha256": hashlib.sha256(
            git_output("status", "--porcelain=v1", "--untracked-files=all").encode()
        ).hexdigest(),
        "checkpoint": str(args.checkpoint.resolve() / "model.pt"),
        "checkpoint_sha256": args.checkpoint_sha256,
        "physical_gpu_index": args.physical_gpu_index,
        "physical_gpu_uuid_nvidia_smi": args.expected_gpu_uuid,
        "physical_gpu_uuid_visible": visible_uuid,
        "seed": args.seed,
        "amp_enabled": amp_enabled,
        "feature_layers": list(FEATURE_LAYERS),
        "hidden_dim": int(model.config.d_model),
        "records": len(entries),
        "tasks": list(actual_tasks),
        "input_manifests": descriptors,
        "input_index_sha256": descriptor_sha256(descriptors),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": arrays_sha,
        "local_checks": local_checks,
        "rows": [
            {**identity, "identity_sha256": identity_hashes[index], "array_index": index}
            for index, identity in enumerate(identities)
        ],
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "records": len(entries), "tasks": actual_tasks, "arrays_sha256": arrays_sha}, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
