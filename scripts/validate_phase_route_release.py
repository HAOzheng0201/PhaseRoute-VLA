#!/usr/bin/env python3
"""Preflight the reproducible A1 + RP-PEP release without loading 34 GB weights."""

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

from a1.vla.dynamic_compute.release import validate_rp_pep_release  # noqa: E402


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
    "a1.vla.dynamic_compute.productive_exit",
    "libero.libero.benchmark",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="also require exactly one visible CUDA device",
    )
    parser.add_argument(
        "--physical-gpu-index",
        type=int,
        help="audit label; only physical GPUs 0-3 are permitted",
    )
    parser.add_argument(
        "--expected-gpu-uuid",
        help="GPU UUID selected by the launcher; required with --require-cuda",
    )
    return parser.parse_args()


def _git(*args: str, root: Path) -> str | None:
    """Return optional provenance without making artifact checks depend on Git."""

    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _normalize_uuid(value: str) -> str:
    result = str(value).strip().lower()
    return result[4:] if result.startswith("gpu-") else result


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    if args.physical_gpu_index is not None and args.physical_gpu_index not in range(4):
        raise ValueError("only physical GPUs 0-3 are permitted")
    if args.require_cuda and args.physical_gpu_index is None:
        raise ValueError("--physical-gpu-index is required with --require-cuda")
    if args.require_cuda and not args.expected_gpu_uuid:
        raise ValueError("--expected-gpu-uuid is required with --require-cuda")

    package_versions: dict[str, str | None] = {}
    missing_packages = []
    for name in REQUIRED_PACKAGES:
        try:
            package_versions[name] = version(name)
        except PackageNotFoundError:
            package_versions[name] = None
            missing_packages.append(name)

    import_errors: dict[str, str] = {}
    for module in REQUIRED_IMPORTS:
        try:
            import_module(module)
        except Exception as error:  # preflight must report the complete import grid
            import_errors[module] = f"{type(error).__name__}: {error}"

    release = validate_rp_pep_release(root)
    submodule = root / "robot_experiments/libero/LIBERO/libero/libero/benchmark"
    checks: dict[str, bool] = {
        "python_3_10": sys.version_info[:2] == (3, 10),
        "all_required_packages": not missing_packages,
        "all_required_imports": not import_errors,
        "libero_submodule_present": submodule.is_dir(),
        "release_artifacts": release["status"] == "PASS",
        "vla_config_yaml": os.environ.get("VLA_CONFIG_YAML")
        in (None, "libero_simulation.yaml"),
    }

    cuda: dict[str, Any] = {"required": bool(args.require_cuda)}
    if args.require_cuda:
        import torch

        cuda.update(
            {
                "available": bool(torch.cuda.is_available()),
                "visible_device_count": int(torch.cuda.device_count()),
                "torch_cuda_runtime": torch.version.cuda,
            }
        )
        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            properties = torch.cuda.get_device_properties(0)
            cuda.update(
                {
                    "visible_name": properties.name,
                    "visible_uuid": str(properties.uuid),
                }
            )
        checks["cuda_available"] = bool(cuda.get("available"))
        checks["exactly_one_visible_gpu"] = cuda.get("visible_device_count") == 1
        checks["physical_gpu_index_front4"] = args.physical_gpu_index in range(4)
        checks["visible_gpu_uuid_matches_expected"] = _normalize_uuid(
            cuda.get("visible_uuid", "")
        ) == _normalize_uuid(args.expected_gpu_uuid)

    status = "PASS" if all(checks.values()) else "FAIL"
    git_worktree_status = _git("status", "--porcelain", root=root)
    result = {
        "status": status,
        "scope": "phase_route_vla_release_preflight",
        "repo_root": str(root),
        "git_commit": _git("rev-parse", "HEAD", root=root),
        "git_branch": _git("branch", "--show-current", root=root),
        "worktree_dirty": (
            bool(git_worktree_status)
            if git_worktree_status is not None
            else None
        ),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "package_versions": package_versions,
        "missing_packages": missing_packages,
        "import_errors": import_errors,
        "physical_gpu_index": args.physical_gpu_index,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "cuda": cuda,
        "release": release,
        "checks": checks,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        path = args.output.resolve()
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    print(output, end="")
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
