#!/usr/bin/env python3
"""No-episode preflight for a preregistered Stage-9 active arm."""

from __future__ import annotations

import argparse
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.route_first_active_protocol import (  # noqa: E402
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
    load_route_first_active_protocol,
    sha256_file,
)
from a1.vla.dynamic_compute.route_first_runtime import (  # noqa: E402
    load_route_first_active_runtime,
)
from a1.vla.dynamic_compute.v3.release import (  # noqa: E402
    validate_phase_route_v3_release,
)


D9_PROTECTED_SHA256 = {
    "a1/vla/value_net.py": (
        "ec3a860427f32d5837e279eb17eeb28befaee9dd7944d46482173c85e8847dc1"
    ),
    "robot_experiments/libero/exit_vla_utils.py": (
        "e5c88b72199c1354fc7b3f2fa22e056b593ee5cdadf7185cc7d1c09fe768051a"
    ),
    "robot_experiments/libero/eval_libero_early_exit.py": (
        "a4e3b1b49cdaf2021b3cd370d8a1e89c927906e7cbd5f8afdccd5ceb5b1826cd"
    ),
}
REQUIRED_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "numpy",
    "mujoco",
    "robosuite",
    "bddl",
    "gym",
    "libero",
    "dlimp",
)
REQUIRED_IMPORTS = (
    "a1.vla.affordvla",
    "a1.vla.affordvla_early_exit",
    "a1.vla.dynamic_compute.route_first_controller",
    "a1.vla.dynamic_compute.route_first_runtime",
    "libero.libero.benchmark",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _normalize_uuid(value: Any) -> str:
    normalized = str(value).strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _nvidia_smi(*args: str) -> str:
    return subprocess.check_output(
        ["nvidia-smi", *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _physical_gpu_snapshot(index: int) -> dict[str, Any]:
    line = _nvidia_smi(
        "-i",
        str(index),
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    fields = [item.strip() for item in line.split(",")]
    if len(fields) != 6 or int(fields[0]) != index:
        raise RuntimeError("physical GPU query returned an unexpected row")
    return {
        "index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_used_mib": int(fields[3]),
        "memory_total_mib": int(fields[4]),
        "utilization_gpu_percent": int(fields[5]),
    }


def _external_compute_processes(expected_uuid: str) -> list[dict[str, Any]]:
    try:
        output = _nvidia_smi(
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        )
    except subprocess.CalledProcessError:
        return [{"query_error": True}]
    result = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3:
            result.append({"parse_error": line})
            continue
        if _normalize_uuid(fields[0]) != _normalize_uuid(expected_uuid):
            continue
        pid = int(fields[1])
        if pid != os.getpid():
            result.append({"gpu_uuid": fields[0], "pid": pid, "used_memory_mib": int(fields[2])})
    return result


def _write_exclusive(path: Path, text: str) -> None:
    target = path.resolve()
    temporary = target.with_name(target.name + ".incomplete")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if args.physical_gpu_index not in range(8):
        raise ValueError("physical GPU index must be in 0..7")

    package_versions: dict[str, str | None] = {}
    missing_packages = []
    for name in REQUIRED_PACKAGES:
        try:
            package_versions[name] = version(name)
        except PackageNotFoundError:
            package_versions[name] = None
            missing_packages.append(name)
    import_errors = {}
    for module in REQUIRED_IMPORTS:
        try:
            import_module(module)
        except Exception as error:
            import_errors[module] = f"{type(error).__name__}: {error}"

    protocol_error = None
    protocol = None
    runtime_error = None
    runtime = None
    try:
        protocol = load_route_first_active_protocol(args.protocol, root)
        frozen = protocol["frozen_implementation"]
        runtime = load_route_first_active_runtime(
            root / frozen["calibrated_router_path"],
            root / frozen["stage7_holdout_path"],
            root / frozen["v3_context_router_path"],
            root / frozen["phase_checkpoint_path"],
        )
    except Exception as error:
        if protocol is None:
            protocol_error = f"{type(error).__name__}: {error}"
        else:
            runtime_error = f"{type(error).__name__}: {error}"

    release = validate_phase_route_v3_release(
        root,
        checkpoint_dir=checkpoint,
        require_backbone=True,
        validate_payloads=True,
    )
    protected = {
        relative: {
            "expected_sha256": expected,
            "actual_sha256": sha256_file(root / relative),
        }
        for relative, expected in D9_PROTECTED_SHA256.items()
    }

    # Sample contention before creating this process's CUDA context.  A second
    # sample after the smoke test closes the race window as far as preflight can
    # without pretending to reserve a shared GPU.
    physical_before = _physical_gpu_snapshot(args.physical_gpu_index)
    external_before = _external_compute_processes(args.expected_gpu_uuid)

    import torch

    cuda: dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "visible_device_count": int(torch.cuda.device_count()),
        "torch_cuda_runtime": torch.version.cuda,
    }
    cuda_smoke_ok = False
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        properties = torch.cuda.get_device_properties(0)
        cuda.update(
            {
                "visible_name": properties.name,
                "visible_uuid": str(properties.uuid),
                "visible_total_memory_bytes": int(properties.total_memory),
            }
        )
        try:
            matrix = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
            output = matrix @ matrix
            torch.cuda.synchronize()
            cuda_smoke_ok = bool(torch.isfinite(output).all())
            del matrix, output
        except Exception as error:
            cuda["smoke_error"] = f"{type(error).__name__}: {error}"

    physical_after = _physical_gpu_snapshot(args.physical_gpu_index)
    external_after = _external_compute_processes(args.expected_gpu_uuid)
    external_processes = external_before + external_after
    worktree_status = _git(root, "status", "--porcelain")
    stage8_commit = (
        protocol.get("frozen_implementation", {}).get("stage8_commit")
        if isinstance(protocol, dict)
        else None
    )
    stage8_ancestor = False
    if isinstance(stage8_commit, str):
        try:
            subprocess.check_call(
                ["git", "merge-base", "--is-ancestor", stage8_commit, "HEAD"],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            stage8_ancestor = True
        except (OSError, subprocess.CalledProcessError):
            pass

    checks = {
        "python_3_10": sys.version_info[:2] == (3, 10),
        "required_packages": not missing_packages,
        "required_imports": not import_errors,
        "protocol_exact": protocol is not None and protocol_error is None,
        "route_first_runtime_loads": runtime is not None and runtime_error is None,
        "stage8_commit_is_ancestor": stage8_ancestor,
        "v3_release_and_backbone": release.get("status") == "PASS",
        "d9_protected_bytes": all(
            item["actual_sha256"] == item["expected_sha256"]
            for item in protected.values()
        ),
        "worktree_clean": worktree_status == "",
        "cuda_available": cuda.get("available") is True,
        "exactly_one_visible_gpu": cuda.get("visible_device_count") == 1,
        "visible_uuid_matches": _normalize_uuid(cuda.get("visible_uuid"))
        == _normalize_uuid(args.expected_gpu_uuid),
        "physical_uuid_matches": _normalize_uuid(physical_before.get("uuid"))
        == _normalize_uuid(args.expected_gpu_uuid)
        == _normalize_uuid(physical_after.get("uuid")),
        "cuda_bfloat16_smoke": cuda_smoke_ok,
        "no_external_compute_process": not external_processes,
        "gpu_idle_at_preflight": physical_before["utilization_gpu_percent"] <= 5,
        "sufficient_free_memory_mib": (
            physical_before["memory_total_mib"]
            - physical_before["memory_used_mib"]
            >= 40_000
        ),
        "simulator_episode_not_opened": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "schema_version": "phase-route-vla.route-first-active-preflight.v1",
        "status": status,
        "scope": "route_first_stage9_active_preflight",
        "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "simulator_episode_opened": False,
        "research_simulation_only": True,
        "deployment_authorized": False,
        "repo_root": str(root),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_branch": _git(root, "branch", "--show-current"),
        "worktree_dirty": worktree_status != "" if worktree_status is not None else None,
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "package_versions": package_versions,
        "missing_packages": missing_packages,
        "import_errors": import_errors,
        "physical_gpu_index": args.physical_gpu_index,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "physical_gpu_before_cuda_smoke": physical_before,
        "physical_gpu_after_cuda_smoke": physical_after,
        "external_compute_processes": external_processes,
        "cuda": cuda,
        "protocol_error": protocol_error,
        "runtime_error": runtime_error,
        "route_first_threshold13": (
            runtime.adapter.threshold13 if runtime is not None else None
        ),
        "d9_protected_files": protected,
        "release": release,
        "checks": checks,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_exclusive(args.output, output)
    print(output, end="")
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
