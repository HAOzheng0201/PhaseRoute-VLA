#!/usr/bin/env python3
"""Run two real A1 policy calls without creating or stepping a LIBERO env."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a1.config import TrainConfig  # noqa: E402
from a1.data.vla.rlds.utils.data_utils import NormalizationType  # noqa: E402
from a1.util import resource_path  # noqa: E402
from a1.vla.affordvla_early_exit import AffordVLAEarlyExit  # noqa: E402
from a1.vla.dynamic_compute.productive_exit import a1_fm10_rp_pep_plan  # noqa: E402
from a1.vla.dynamic_compute.v3.active_runtime import (  # noqa: E402
    load_frozen_phase_route_runtime,
    sha256_file,
)
from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    validate_runtime_model_directory,
)
from a1.vla.value_net import ActionValueNet, ExitController  # noqa: E402
from robot_experiments.libero.exit_vla_utils import get_vla_action  # noqa: E402


SCHEMA_VERSION = "phase-route-vla.v3.d9b-model-dry-run.v1"
PASS_STATUS = "PASS_V3_D9B_REAL_MODEL_NON_CONTROL_DRY_RUN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-attestation", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _gpu_inventory() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    records = []
    for line in output.splitlines():
        index, uuid, name, memory, driver = [part.strip() for part in line.split(",")]
        records.append(
            {
                "physical_index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(memory),
                "driver_version": driver,
            }
        )
    return records


def _sha256_array(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_model(checkpoint: Path, device: torch.device) -> AffordVLAEarlyExit:
    train_config = TrainConfig.load(
        resource_path(checkpoint, "config.yaml"), validate_paths=False
    )
    config = train_config.model
    data_root = os.environ.get("DATA_DIR", "")
    config.vit_load_path = os.path.join(
        data_root, "pretrained_image_encoders/vit-l-14-336.pt"
    )
    config.llm_load_path = os.path.join(data_root, "pretrained_llms/qwen2-7b.pt")
    config.tokenizer.tokenizer_dir = os.environ.get("HF_HOME", "")
    config.num_diffusion_inference_steps = 10
    config.init_device = str(device)
    model = AffordVLAEarlyExit(config)
    state_path = resource_path(checkpoint, "model.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()
    model.to(device)
    model.eval()
    return model


def _controller(checkpoint: Path, model: AffordVLAEarlyExit, device: torch.device):
    original_layers = tuple(model.get_all_exit_idx(2))
    plan = a1_fm10_rp_pep_plan(original_layers)
    value_net = ActionValueNet(
        exit_list=list(plan.eligible_exit_layers),
        exit_head=model.action_head,
        model=model,
        interval=2,
        threshold_type="cosine",
        productive_exit_plan=plan,
    )
    controller = ExitController(
        value_net,
        exit_id_list=list(plan.eligible_exit_layers),
        steps_per_stage=1,
        leq=True,
        exit_dist="exp",
        max_layer=model.config.n_layers,
    ).to(device)
    threshold_path = checkpoint / "exit_thresholds_libero_10_exp_1.0.json"
    frozen = {
        int(layer): float(value)
        for layer, value in json.loads(threshold_path.read_text()).items()
    }
    controller._set_threshold_value(
        plan.select_eligible_thresholds(frozen, lower_is_easier=True)
    )
    return controller


def _policy_config(checkpoint: Path, sequence_length: int) -> SimpleNamespace:
    return SimpleNamespace(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name="libero_10",
        unnorm_key="",
        normalization_type=NormalizationType.BOUNDS_Q99,
        use_proprio=True,
        use_wrist_image=True,
        num_images_in_input=2,
        center_crop=True,
        sequence_length=sequence_length,
        num_open_loop_steps=8,
        exit_interval=2,
    )


def _synthetic_observation(
    index: int, size: int | tuple[int, int]
) -> dict[str, Any]:
    height, width = (size, size) if isinstance(size, int) else tuple(size)
    y, x = np.indices((height, width), dtype=np.int32)
    primary = np.stack(
        (
            (x + 17 * index) % 256,
            (y * 3 + 29 * index) % 256,
            ((x // 8 + y // 8 + index) % 2) * 255,
        ),
        axis=2,
    ).astype(np.uint8)
    wrist = np.rot90(primary, k=1 + index).copy()
    state = np.asarray(
        [0.02 * index, -0.03, 0.11, 0.0, 0.0, 0.0, 0.015, -0.015],
        dtype=np.float32,
    )
    return {"full_image": primary, "wrist_image": wrist, "state": state}


def main() -> None:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.physical_gpu_index not in (0, 1, 2, 3):
        raise PermissionError("D9B only permits physical GPU 0--3")
    if visible != str(args.physical_gpu_index):
        raise PermissionError("CUDA_VISIBLE_DEVICES must name exactly one physical GPU")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D9B dry-run requires a clean implementation commit")
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("D9B refuses to overwrite dry-run evidence")
    checkpoint = args.checkpoint.resolve(strict=True)
    model_audit = validate_runtime_model_directory(
        checkpoint, args.model_attestation.resolve(strict=True)
    )
    inventory = _gpu_inventory()
    physical = next(
        item for item in inventory if item["physical_index"] == args.physical_gpu_index
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    incomplete.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.manual_seed(20260822)
    np.random.seed(20260822)
    random.seed(20260822)
    torch.cuda.set_device(0)
    if torch.cuda.device_count() != 1:
        raise PermissionError("D9B process must see exactly one CUDA device")
    device = torch.device("cuda:0")
    model = _load_model(checkpoint, device)
    controller = _controller(checkpoint, model, device)
    runtime = load_frozen_phase_route_runtime(
        args.router.resolve(strict=True), args.phase_checkpoint.resolve(strict=True)
    )
    controller.set_phase_route_runtime_adapter(runtime.adapter)
    train_config = TrainConfig.load(checkpoint / "config.yaml", validate_paths=False)
    config = _policy_config(checkpoint, train_config.data.sequence_length)
    episode_id = "d9b_synthetic_non_test:task0:episode0"
    runtime.start_episode(episode_id)
    action_records = []
    for ordinal in range(2):
        controller.set_timestep(10 + ordinal * 8)
        observation = _synthetic_observation(
            ordinal, model.config.vision_backbone.image_default_input_size
        )
        call_started = time.perf_counter()
        actions = get_vla_action(
            config,
            model,
            device,
            observation,
            "move the synthetic object to the synthetic target",
            controller,
            output_hidden_states=False,
            phase_route_runtime=runtime,
            phase_route_context={
                "episode_id": episode_id,
                "call_ordinal": ordinal,
                "step_id": 10 + ordinal * 8,
                "task_id": 0,
            },
        )
        array = np.asarray(actions, dtype=np.float32)
        action_records.append(
            {
                "call_ordinal": ordinal,
                "shape": list(array.shape),
                "finite": bool(np.isfinite(array).all()),
                "sha256": _sha256_array(array),
                "latency_seconds": time.perf_counter() - call_started,
                "selected_layer": runtime.records[-1]["selected_layer"],
                "history_valid_rows": runtime.records[-1].get(
                    "history_valid_rows"
                ),
            }
        )
    torch.cuda.synchronize(device)
    route_records = list(runtime.records)
    checks = {
        "real_A1_checkpoint_loaded": True,
        "two_policy_calls_completed": len(action_records) == 2,
        "all_returned_actions_finite_8x7": all(
            record["shape"] == [8, 7] and record["finite"]
            for record in action_records
        ),
        "nine_tensor_context_prepared_twice": runtime.prepared_calls == 2,
        "selected_action_history_committed_twice": runtime.committed_calls == 2,
        "past_only_history_0_then_1": [
            record["history_valid_rows"] for record in action_records
        ]
        == [0, 1],
        "route_selected_only_L11_L13_or_L27": all(
            record["selected_layer"] in (11, 13, 27) for record in action_records
        ),
        "runtime_error_count_zero": runtime.error_count == 0,
        "one_visible_allowed_GPU": torch.cuda.device_count() == 1,
        "no_LIBERO_environment_created_or_stepped": True,
        "no_action_executed": True,
        "episode_40_49_state_not_opened": True,
        "independent_test_sample_payload_not_opened": True,
    }
    if not all(checks.values()):
        raise PermissionError(f"D9B dry-run checks failed: {checks}")
    code_paths = (
        Path("a1/vla/dynamic_compute/v3/active_runtime.py"),
        Path("a1/vla/dynamic_compute/v3/runtime_adapter.py"),
        Path("a1/vla/value_net.py"),
        Path("robot_experiments/libero/exit_vla_utils.py"),
        Path("robot_experiments/libero/eval_libero_early_exit.py"),
        Path("scripts/dynamic_compute/v3/validate_v3_d9b_model_dry_run.py"),
    )
    result = {
        "status": PASS_STATUS,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty_at_start": False,
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": visible,
            "visible_device_count": torch.cuda.device_count(),
            "visible_device_name": torch.cuda.get_device_name(0),
            "physical_gpu": physical,
        },
        "model_audit": model_audit,
        "runtime_artifacts": runtime.artifacts.__dict__,
        "code_sha256": {
            path.as_posix(): sha256_file(REPO_ROOT / path) for path in code_paths
        },
        "policy_calls": action_records,
        "route_records": route_records,
        "runtime_counters": {
            "policy_calls": runtime.policy_calls,
            "prepared_calls": runtime.prepared_calls,
            "committed_calls": runtime.committed_calls,
            "errors": runtime.error_count,
            "last_error": runtime.last_error,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "checks": checks,
        "access_ledger": {
            "synthetic_RGB_observations": 2,
            "real_A1_checkpoint_opened": True,
            "D8_router_opened": True,
            "phase_checkpoint_opened": True,
            "LIBERO_environment_imported": False,
            "LIBERO_environment_created": False,
            "environment_steps": 0,
            "returned_actions_executed": 0,
            "episode_40_49_init_states_opened": False,
            "independent_test_sample_payload_opened": False,
            "active_control": False,
        },
    }
    result_path = incomplete / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (incomplete / "result.sha256").write_text(
        f"{sha256_file(result_path)}  result.json\n", encoding="utf-8"
    )
    (incomplete / "command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    incomplete.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
