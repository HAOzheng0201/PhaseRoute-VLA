#!/usr/bin/env python3
"""Supervise three isolated Stage 10 arm processes on one physical GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._route_first_stage10_contracts import ACTIVE, CONTRACT  # noqa: E402


sha256_file = CONTRACT.sha256_file
validate_local_state_artifacts = CONTRACT.validate_local_state_artifacts
ACTIVE_TRIPLET_SCHEMA = ACTIVE.ACTIVE_TRIPLET_SCHEMA
ARM_ATTESTATION_SCHEMA = ACTIVE.ARM_ATTESTATION_SCHEMA
Stage10ActiveError = ACTIVE.Stage10ActiveError
expected_triplet_directory = ACTIVE.expected_triplet_directory
load_runner_readiness = ACTIVE.load_runner_readiness
normalize_gpu_uuid = ACTIVE.normalize_gpu_uuid
select_triplet = ACTIVE.select_triplet
validate_triplet_record = ACTIVE.validate_triplet_record


PREFLIGHT = REPO_ROOT / "scripts/validate_route_first_stage10_preflight.py"
RUNNER = REPO_ROOT / "scripts/run_route_first_stage10_arm.py"
POSTFLIGHT = REPO_ROOT / "scripts/validate_route_first_stage10_postflight.py"
VALIDATOR = REPO_ROOT / "scripts/validate_route_first_stage10_arm.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--replicate-id", type=int, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "model/libero_exit"
    )
    parser.add_argument(
        "--python-bin", type=Path, default=Path(sys.executable)
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Stage10ActiveError(f"JSON object required: {path}")
    return dict(value)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _gpu_uuid(index: int) -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def _attempt_directory(root: Path, position: int, method: str) -> Path:
    attempts = root / ".attempts" / f"arm{position}_{method}"
    attempts.mkdir(parents=True, exist_ok=True)
    ordinal = 1 + len(list(attempts.iterdir()))
    target = attempts / f"attempt_{ordinal:03d}.incomplete"
    target.mkdir(exist_ok=False)
    return target


def _command_text(environment: Mapping[str, str], command: list[str]) -> str:
    return shlex.join(["env", *[f"{k}={v}" for k, v in environment.items()], *command]) + "\n"


def _run_logged(
    command: list[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    echo: bool = False,
) -> subprocess.CompletedProcess[str]:
    with log_path.open("x", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, **environment},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            output.flush()
            if echo:
                print(line, end="", flush=True)
        return subprocess.CompletedProcess(command, process.wait())


def _abort(
    attempt: Path,
    *,
    spec: Any,
    method: str,
    position: int,
    source_commit: str,
    gpu_index: int,
    gpu_uuid: str,
    state_sha256: str,
    stage: str,
    returncode: int,
) -> None:
    _write_json(
        attempt / "abort.json",
        {
            "status": "ABORT_ROUTE_FIRST_STAGE10_INFRASTRUCTURE_FAILURE",
            "timestamp_utc": utc_now(),
            "failure_stage": stage,
            "returncode": returncode,
            "task_id": spec.task_id,
            "replicate_id": spec.replicate_id,
            "cluster_key": spec.cluster_key,
            "method": method,
            "arm_position": position,
            "arm_order": list(spec.arm_order),
            "state_seed": spec.state_seed,
            "state_sha256": state_sha256,
            "policy_seed": spec.policy_seed,
            "source_git_commit": source_commit,
            "physical_gpu_index": gpu_index,
            "gpu_uuid": gpu_uuid,
            "same_tuple_required_for_retry": True,
            "valid_task_failure": False,
            "outcome_based_retry": False,
            "replacement_state_or_seed": False,
        },
    )


def _completed_arm(
    path: Path, *, spec: Any, method: str, position: int, commit: str, uuid: str
) -> dict[str, Any]:
    attestation_path = path / "arm_attestation.json"
    sidecar_path = path / "arm_attestation.sha256"
    if not attestation_path.is_file() or not sidecar_path.is_file():
        raise Stage10ActiveError(f"completed arm is not sealed: {path}")
    attestation = _object(attestation_path)
    digest = sha256_file(attestation_path)
    if sidecar_path.read_text(encoding="utf-8").split()[0] != digest:
        raise Stage10ActiveError("arm attestation sidecar differs")
    if (
        attestation.get("schema_version") != ARM_ATTESTATION_SCHEMA
        or attestation.get("status") != "PASS"
        or attestation.get("task_id") != spec.task_id
        or attestation.get("replicate_id") != spec.replicate_id
        or attestation.get("cluster_key") != spec.cluster_key
        or attestation.get("method") != method
        or attestation.get("arm_position") != position
        or attestation.get("policy_seed") != spec.policy_seed
        or attestation.get("source_git_commit") != commit
        or normalize_gpu_uuid(attestation.get("gpu_uuid")) != normalize_gpu_uuid(uuid)
        or not all(attestation.get("checks", {}).values())
    ):
        raise Stage10ActiveError("completed arm attestation identity differs")
    return {
        "method": method,
        "arm_position": position,
        "success": attestation["success"],
        "environment_steps": attestation["environment_steps"],
        "policy_calls": attestation["policy_calls"],
        "selected_layer_counts": attestation["selected_layer_counts"],
        "route_exactly_one_fm_calls": attestation[
            "route_exactly_one_fm_calls"
        ],
        "policy_p50_ms": attestation["policy_p50_ms"],
        "policy_seed": spec.policy_seed,
        "state_sha256": attestation["state_sha256"],
        "source_git_commit": commit,
        "gpu_uuid": uuid,
        "evidence_valid": True,
        "attestation_path": attestation_path.relative_to(path.parent.parent).as_posix(),
        "attestation_sha256": digest,
    }


def _run(args: argparse.Namespace) -> None:
    if args.physical_gpu_index not in range(8):
        raise ValueError("physical GPU index must be in 0..7")
    if _git("status", "--porcelain=v1"):
        raise PermissionError("Stage 10 triplet requires a clean worktree")
    readiness = load_runner_readiness(REPO_ROOT)
    source_commit = _git("rev-parse", "HEAD")
    spec = select_triplet(REPO_ROOT, args.task_id, args.replicate_id)
    output = expected_triplet_directory(
        REPO_ROOT, spec.task_id, spec.replicate_id
    )
    if output.exists() and not args.resume:
        raise FileExistsError("triplet output exists; use --resume")
    if not output.exists() and args.resume:
        raise FileNotFoundError("--resume requires an existing triplet output")
    output.mkdir(parents=True, exist_ok=args.resume)
    uuid = args.expected_gpu_uuid or _gpu_uuid(args.physical_gpu_index)
    if normalize_gpu_uuid(_gpu_uuid(args.physical_gpu_index)) != normalize_gpu_uuid(
        uuid
    ):
        raise Stage10ActiveError("physical GPU UUID differs")
    local = validate_local_state_artifacts(REPO_ROOT)
    state_records = [
        item
        for item in local["attestation"]["records"]
        if item["task_id"] == spec.task_id
        and item["replicate_id"] == spec.replicate_id
    ]
    if len(state_records) != 1:
        raise Stage10ActiveError("triplet state attestation differs")
    state_sha256 = state_records[0]["state_sha256"]
    python = str(args.python_bin.resolve(strict=True))
    checkpoint = str(args.checkpoint.resolve(strict=True))
    environment = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": uuid,
        "MUJOCO_EGL_DEVICE_ID": "0",
        "DATA_DIR": str(REPO_ROOT),
        "HF_HOME": os.environ.get("HF_HOME", str(REPO_ROOT / ".cache/huggingface")),
        "LIBERO_CONFIG_PATH": os.environ.get(
            "LIBERO_CONFIG_PATH", str(REPO_ROOT / ".cache/libero")
        ),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        "VLA_CONFIG_YAML": "libero_simulation.yaml",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "PYTHONNOUSERSITE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    command_path = output / "command.txt"
    if not command_path.exists():
        command_path.write_text(
            _command_text(environment, [python, *sys.argv]), encoding="utf-8"
        )
    arms: dict[str, Any] = {}
    for position, method in enumerate(spec.arm_order, start=1):
        final = output / f"arm{position}_{method}"
        if final.is_dir():
            arms[method] = _completed_arm(
                final,
                spec=spec,
                method=method,
                position=position,
                commit=source_commit,
                uuid=uuid,
            )
            print(f"[Stage10] arm already sealed: {method}", flush=True)
            continue
        if final.exists():
            raise Stage10ActiveError("arm output is not a directory")
        attempt = _attempt_directory(output, position, method)
        preflight_command = [
            python,
            str(PREFLIGHT),
            "--task-id",
            str(spec.task_id),
            "--replicate-id",
            str(spec.replicate_id),
            "--method",
            method,
            "--arm-position",
            str(position),
            "--physical-gpu-index",
            str(args.physical_gpu_index),
            "--expected-gpu-uuid",
            uuid,
            "--output",
            str(attempt / "preflight.json"),
        ]
        runner_command = [
            python,
            str(RUNNER),
            "--task-id",
            str(spec.task_id),
            "--replicate-id",
            str(spec.replicate_id),
            "--method",
            method,
            "--arm-position",
            str(position),
            "--physical-gpu-index",
            str(args.physical_gpu_index),
            "--expected-gpu-uuid",
            uuid,
            "--checkpoint",
            checkpoint,
            "--preflight",
            str(attempt / "preflight.json"),
            "--output-dir",
            str(attempt),
        ]
        (attempt / "command.txt").write_text(
            _command_text(environment, runner_command), encoding="utf-8"
        )
        print(
            f"[Stage10] preflight task={spec.task_id} replicate={spec.replicate_id} "
            f"arm={position} method={method} GPU={args.physical_gpu_index}",
            flush=True,
        )
        preflight = _run_logged(
            preflight_command,
            environment=environment,
            log_path=attempt / "preflight_stdout.log",
        )
        if preflight.returncode:
            _abort(
                attempt,
                spec=spec,
                method=method,
                position=position,
                source_commit=source_commit,
                gpu_index=args.physical_gpu_index,
                gpu_uuid=uuid,
                state_sha256=state_sha256,
                stage="preflight",
                returncode=preflight.returncode,
            )
            raise RuntimeError("Stage 10 arm preflight failed")
        print(f"[Stage10] launching isolated arm: {method}", flush=True)
        print("[Stage10] command: " + shlex.join(runner_command), flush=True)
        arm_process = _run_logged(
            runner_command,
            environment=environment,
            log_path=attempt / "stdout.log",
            echo=True,
        )
        postflight_command = [
            python,
            str(POSTFLIGHT),
            "--physical-gpu-index",
            str(args.physical_gpu_index),
            "--expected-gpu-uuid",
            uuid,
            "--output",
            str(attempt / "gpu_postflight.json"),
        ]
        postflight = subprocess.run(
            postflight_command,
            cwd=REPO_ROOT,
            env={**os.environ, **environment},
            capture_output=True,
            text=True,
            check=False,
        )
        if arm_process.returncode or postflight.returncode:
            _abort(
                attempt,
                spec=spec,
                method=method,
                position=position,
                source_commit=source_commit,
                gpu_index=args.physical_gpu_index,
                gpu_uuid=uuid,
                state_sha256=state_sha256,
                stage=("arm_process" if arm_process.returncode else "postflight"),
                returncode=arm_process.returncode or postflight.returncode,
            )
            raise RuntimeError("Stage 10 arm process or postflight failed")
        validator_command = [
            python,
            str(VALIDATOR),
            str(attempt),
            "--task-id",
            str(spec.task_id),
            "--replicate-id",
            str(spec.replicate_id),
            "--method",
            method,
            "--arm-position",
            str(position),
            "--physical-gpu-index",
            str(args.physical_gpu_index),
            "--expected-gpu-uuid",
            uuid,
        ]
        validation = subprocess.run(
            validator_command,
            cwd=REPO_ROOT,
            env={**os.environ, **environment},
            capture_output=True,
            text=True,
            check=False,
        )
        if validation.returncode:
            _abort(
                attempt,
                spec=spec,
                method=method,
                position=position,
                source_commit=source_commit,
                gpu_index=args.physical_gpu_index,
                gpu_uuid=uuid,
                state_sha256=state_sha256,
                stage="evidence_validation",
                returncode=validation.returncode,
            )
            raise RuntimeError(
                "Stage 10 arm evidence validation failed: "
                + validation.stderr[-2000:]
            )
        attempt.replace(final)
        arms[method] = _completed_arm(
            final,
            spec=spec,
            method=method,
            position=position,
            commit=source_commit,
            uuid=uuid,
        )
        print(
            f"[Stage10] sealed arm={method} raw_success={arms[method]['success']}",
            flush=True,
        )
    if len(arms) != 3:
        raise Stage10ActiveError("triplet did not complete all three arms")
    triplet = {
        "schema_version": ACTIVE_TRIPLET_SCHEMA,
        "status": "COMPLETE_ROUTE_FIRST_STAGE10_TRIPLET",
        "timestamp_utc": utc_now(),
        "task_id": spec.task_id,
        "replicate_id": spec.replicate_id,
        "cluster_key": spec.cluster_key,
        "state_seed": spec.state_seed,
        "policy_seed": spec.policy_seed,
        "state_sha256": state_sha256,
        "arm_order": list(spec.arm_order),
        "source_git_commit": source_commit,
        "physical_gpu_index": args.physical_gpu_index,
        "gpu_uuid": uuid,
        "runner_readiness_sha256": sha256_file(
            REPO_ROOT
            / "results/route_first/route_first_stage10_runner_readiness.json"
        ),
        "arms": arms,
        "infrastructure_attempt_directories": len(
            list((output / ".attempts").glob("*/*"))
        ),
        "retry_policy": {
            "valid_task_failure_retained": True,
            "outcome_based_retry": False,
            "replacement_state_or_seed": False,
        },
        "claim_boundary": {
            "raw_triplet_evidence_only": True,
            "cross_triplet_aggregate_computed": False,
            "stage10_gate_evaluated": False,
        },
    }
    validate_triplet_record(triplet, spec=spec)
    result_path = output / "triplet_record.json"
    sidecar = output / "triplet_record.sha256"
    if result_path.exists() or sidecar.exists():
        if not args.resume:
            raise FileExistsError("triplet record already exists")
        existing = _object(result_path)
        validate_triplet_record(existing, spec=spec)
        if sidecar.read_text(encoding="utf-8").split()[0] != sha256_file(
            result_path
        ):
            raise Stage10ActiveError("existing triplet record SHA differs")
    else:
        _write_json(result_path, triplet)
        sidecar.write_text(
            f"{sha256_file(result_path)}  triplet_record.json\n",
            encoding="utf-8",
        )
    print(
        f"[Stage10] triplet complete task={spec.task_id} "
        f"replicate={spec.replicate_id}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    _run(args)


if __name__ == "__main__":
    main()
