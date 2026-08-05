"""Audit whether A1's ActionValueNet history survives policy-call boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--cache-record", type=Path, required=True)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _optional(value: np.ndarray, *, device: torch.device, dtype: torch.dtype):
    if value.size == 0:
        return None
    return torch.from_numpy(value).to(device=device, dtype=dtype).unsqueeze(0)


def _sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.fm_steps < 1:
        raise ValueError("fm-steps must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("audit requires exactly one visible CUDA device")

    from robot_experiments.libero.eval_libero_early_exit import (
        GenerateConfig,
        initialize_and_load_model,
    )
    from scripts.dynamic_compute.smoke_m1_telemetry import make_exit_controller

    checkpoint = args.checkpoint.resolve()
    cfg = GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name="libero_spatial",
        action_head_flow_matching_inference_steps=args.fm_steps,
        exit_interval=2,
        steps_per_stage=1,
        threshold_type="cosine",
        exit_dist="exp",
        exit_ratio=1.0,
        use_wandb=False,
        save_rollout_video=False,
    )
    model, device, _ = initialize_and_load_model(cfg)
    controller = make_exit_controller(cfg, model, device)

    with np.load(args.cache_record) as shard:
        arrays = {name: shard[name].copy() for name in shard.files}
    input_ids = torch.from_numpy(arrays["input_ids"]).to(
        device=device, dtype=torch.int64
    ).unsqueeze(0)
    projected = torch.from_numpy(arrays["projected_features"]).to(
        device=device, dtype=model.transformer.wte.embedding.dtype
    ).unsqueeze(0)
    image_input_idx = torch.from_numpy(arrays["image_input_idx"]).to(
        device=device, dtype=torch.int64
    ).unsqueeze(0)
    response_mask = _optional(
        arrays["response_mask"], device=device, dtype=torch.bool
    )
    attention_mask = _optional(
        arrays["attention_mask"], device=device, dtype=torch.bool
    )
    attention_bias = _optional(
        arrays["attention_bias"], device=device, dtype=torch.float32
    )
    subsegment_ids = _optional(
        arrays["subsegment_ids"], device=device, dtype=torch.int64
    )
    position_ids = _optional(
        arrays["position_ids"], device=device, dtype=torch.int64
    )
    action_proprio = torch.from_numpy(arrays["action_proprio"]).to(
        device=device, dtype=torch.float32
    )
    proprio_token_idx = torch.from_numpy(arrays["proprio_token_idx"]).to(
        device=device, dtype=torch.int64
    )
    teacher_action = torch.from_numpy(arrays["teacher_normalized_action"]).to(
        device=device, dtype=torch.float32
    ).unsqueeze(0)
    cpu_rng_state = torch.from_numpy(arrays["cpu_rng_state"])
    cuda_rng_state = torch.from_numpy(arrays["cuda_rng_state"])

    rows: list[dict[str, Any]] = []
    for call_index in range(2):
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device=device)
        controller.set_timestep(call_index)
        traces = []

        def trace_callback(payload):
            traces.append(dict(payload))

        before = len(controller.value_net.action_list)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                response_mask=response_mask,
                image_input_idx=image_input_idx,
                subsegment_ids=subsegment_ids,
                position_ids=position_ids,
                action_proprio=action_proprio,
                proprio_token_idx=proprio_token_idx,
                output_hidden_states=False,
                use_cache=True,
                precomputed_projected_features=projected,
                exit_controller=controller,
                fm_trace_callback=trace_callback,
            )
        torch.cuda.synchronize(device)
        action = output.exit_action.detach().to(dtype=torch.float32)
        rows.append(
            {
                "call_index": call_index,
                "history_length_before": before,
                "history_length_after": len(controller.value_net.action_list),
                "exit_layer": int(output.exit_layer),
                "trace_count": len(traces),
                "trace_layers": [int(trace["candidate_layer"]) for trace in traces],
                "trace_roles": [str(trace["candidate_role"]) for trace in traces],
                "action_sha256": _sha256_tensor(action),
                "teacher_max_abs_error": float((action - teacher_action).abs().max()),
            }
        )

    result = {
        "status": "PASS",
        "scope": "m420b_action_value_history_audit",
        "checkpoint": str(checkpoint / "model.pt"),
        "cache_record": str(args.cache_record.resolve()),
        "fm_steps": args.fm_steps,
        "physical_gpu_uuid_visible": str(torch.cuda.get_device_properties(0).uuid),
        "rows": rows,
        "history_persists_across_calls": rows[1]["history_length_before"] > 0,
        "second_call_has_comparison_trace": "comparison_previous"
        in rows[1]["trace_roles"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
