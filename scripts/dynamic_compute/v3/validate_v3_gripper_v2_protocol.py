#!/usr/bin/env python3
"""Freeze and validate the V3-D1 Gripper-v2 design-only protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO_ROOT = Path(__file__).absolute().parents[3]
MODULE_DIRECTORY = REPO_ROOT / "a1" / "vla" / "dynamic_compute" / "v3"
sys.path.insert(0, str(MODULE_DIRECTORY))

import gripper_v2_protocol as gp  # noqa: E402


DEFAULT_PROTOCOL = "configs/research/v3/gripper_v2/protocol.json"
DEFAULT_OUTPUT = "results/v3/v3_d1_gripper_v2_protocol_validation.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--legacy-source-root", type=Path, required=True)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--initialize-default-protocol",
        action="store_true",
        help="create the frozen protocol before validating it",
    )
    return parser.parse_args(argv)


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite frozen output: {destination}")
    temporary = destination.with_name(destination.name + ".incomplete")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"incomplete output already exists: {temporary}")
    payload = (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite frozen output: {destination}"
        ) from error
    os.unlink(temporary)


def _relative_path(value: str | Path, *, context: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise gp.ProtocolPathError(f"{context} must be a contained relative path")
    return path


def _load_json(path: Path, *, maximum: int, context: str) -> Any:
    return gp.decode_json_bytes(
        gp.read_bounded_regular_file(path, maximum=maximum), context=context
    )


def _verify_sha(path: Path, expected: str, *, context: str) -> str:
    observed = gp.sha256_file(path)
    if observed != expected:
        raise gp.ProtocolSchemaError(
            f"{context} SHA-256 differs: {observed} != {expected}"
        )
    return observed


def _portable(path: Path, root: Path) -> str:
    try:
        return path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError:
        return f"<LEGACY_SOURCE>/{path.name}"


def _validate_d0(value: Any, lineage: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise gp.ProtocolSchemaError("D0 result must be an object")
    if value.get("status") != lineage["d0_required_status"]:
        raise gp.ProtocolSchemaError("D0 status does not authorize D1")
    if value.get("candidate_split_authorized") is not True:
        raise gp.ProtocolSchemaError("D0 candidate split is not authorized")
    counts = value.get("counts")
    if not isinstance(counts, dict) or (
        counts.get("used_unique_keys"),
        counts.get("candidate_records"),
        counts.get("candidate_known_used_keys"),
    ) != (604, 380, 0):
        raise gp.ProtocolSchemaError("D0 key counts differ")
    integrity = value.get("input_selection_integrity")
    if not isinstance(integrity, dict) or integrity.get(
        "selection_bundle_sha256"
    ) != lineage["d0_selection_bundle_sha256"]:
        raise gp.ProtocolSchemaError("D0 selection bundle differs")
    process = value.get("process_boundary")
    if process != {
        "cuda_visible_devices": "",
        "forbidden_modules_loaded": [],
        "stdlib_only": True,
    }:
        raise gp.ProtocolSchemaError("D0 process boundary differs")


def _validate_legacy_c355(value: Any, lineage: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise gp.ProtocolSchemaError("legacy C3.55 result must be an object")
    if value.get("status") != lineage["legacy_c355_required_status"]:
        raise gp.ProtocolSchemaError("legacy C3.55 status differs")
    dispositions = value.get("family_dispositions")
    if not isinstance(dispositions, dict) or dispositions.get("gripper") != lineage[
        "legacy_gripper_disposition"
    ]:
        raise gp.ProtocolSchemaError("legacy gripper negative disposition differs")
    gates = value.get("scientific_gates")
    if not isinstance(gates, dict):
        raise gp.ProtocolSchemaError("legacy C3.55 scientific gates are missing")
    magnitude = gates.get("gripper_positive_magnitude")
    family = gates.get("family_pass")
    if (
        not isinstance(magnitude, dict)
        or magnitude.get("pass") is not False
        or magnitude.get("every_layer_ratio_below_one") is not False
        or not isinstance(family, dict)
        or family.get("gripper") is not False
    ):
        raise gp.ProtocolSchemaError("legacy gripper failure is not preserved")
    evaluation = value.get("evaluation")
    try:
        failed = evaluation["metrics"]["gripper_positive_magnitude"]["targets"][
            "gripper_step_mismatch_fraction"
        ]["by_layer"]["13"]
    except (KeyError, TypeError) as error:
        raise gp.ProtocolSchemaError(
            "legacy failed gripper metric is missing"
        ) from error
    expected = lineage["legacy_failed_metric"]
    if (
        failed.get("value") != expected["value"]
        or failed.get("support_count") != expected["positive_support"]
        or failed.get("loss") != "mae"
    ):
        raise gp.ProtocolSchemaError("legacy failed gripper metric differs")


def _synthetic_contract_sha256() -> str:
    teacher = [-1.0] * 8
    cases = {
        "identical": gp.construct_gripper_targets(
            {11: teacher, 13: teacher}, teacher
        ),
        "single_step_vs_all_flip": gp.construct_gripper_targets(
            {
                11: [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                13: [1.0] * 8,
            },
            teacher,
        ),
        "threshold_tie": gp.construct_gripper_targets(
            {11: [0.0] * 8, 13: teacher}, teacher
        ),
    }
    return gp.canonical_json_sha256(cases)


def _fold_contract_sha256() -> str:
    assignments = [
        gp.grouped_fold_assignment(
            task_id=task,
            episode_index=episode,
            candidate_layer=layer,
        )
        for task in range(10)
        for episode in gp.DEVELOPMENT_EPISODES
        for layer in gp.DECISION_LAYERS
    ]
    return gp.canonical_json_sha256(assignments)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.absolute()
    protocol_relative = _relative_path(args.protocol, context="protocol")
    protocol_path = gp.resolve_regular_file(repo_root, protocol_relative)
    protocol = gp.load_protocol(protocol_path)
    lineage = protocol["lineage"]

    bindings: list[dict[str, Any]] = []

    def repo_binding(path_key: str, sha_key: str, context: str) -> Path:
        relative = lineage[path_key]
        path = gp.resolve_regular_file(repo_root, relative)
        observed = _verify_sha(path, lineage[sha_key], context=context)
        bindings.append({"id": context, "path": relative, "sha256": observed})
        return path

    d0_path = repo_binding("d0_result_path", "d0_result_sha256", "D0 result")
    _validate_d0(_load_json(d0_path, maximum=2 * 1024 * 1024, context="D0 result"), lineage)
    repo_binding(
        "legacy_manifest_path",
        "legacy_manifest_sha256",
        "legacy evidence manifest",
    )

    role_results: list[dict[str, Any]] = []
    for role_contract in protocol["data_roles"]:
        selection = gp.resolve_regular_file(
            repo_root, role_contract["selection_path"]
        )
        observed = _verify_sha(
            selection,
            role_contract["selection_sha256"],
            context=f'{role_contract["role"]} selection',
        )
        value = _load_json(
            selection, maximum=1024 * 1024, context=role_contract["role"]
        )
        gp.validate_selection_document(
            value,
            role=role_contract["role"],
            episodes=role_contract["episode_indices"],
            expected_count=role_contract["key_count"],
        )
        role_results.append(
            {
                "role": role_contract["role"],
                "key_count": role_contract["key_count"],
                "selection_sha256": observed,
                "first_label_access_stage": role_contract[
                    "first_label_access_stage"
                ],
            }
        )
        bindings.append(
            {
                "id": f'{role_contract["role"]} selection',
                "path": role_contract["selection_path"],
                "sha256": observed,
            }
        )

    legacy_root = args.legacy_source_root.absolute()
    legacy_path = gp.resolve_regular_file(
        legacy_root, lineage["legacy_c355_result_path"]
    )
    legacy_sha = _verify_sha(
        legacy_path,
        lineage["legacy_c355_result_sha256"],
        context="legacy C3.55 negative result",
    )
    _validate_legacy_c355(
        _load_json(
            legacy_path, maximum=2 * 1024 * 1024, context="legacy C3.55 result"
        ),
        lineage,
    )
    bindings.append(
        {
            "id": "legacy C3.55 negative result",
            "path": f'<LEGACY_SOURCE>/{lineage["legacy_c355_result_path"]}',
            "sha256": legacy_sha,
        }
    )

    forbidden_loaded = sorted(
        name
        for name in ("torch", "numpy", "tensorflow", "jax")
        if name in sys.modules
    )
    if forbidden_loaded:
        raise gp.ProtocolSchemaError("ML modules were loaded in D1 validator")
    protocol_sha = gp.sha256_file(protocol_path)
    return {
        "schema_version": "phase-route-vla.v3.gripper-v2-protocol-validation.v1",
        "stage": "V3-D1",
        "status": "PASS_D1_GRIPPER_V2_PROTOCOL_FROZEN",
        "protocol": {
            "path": protocol_relative.as_posix(),
            "id": protocol["protocol_id"],
            "schema_version": protocol["schema_version"],
            "raw_sha256": protocol_sha,
            "canonical_sha256": gp.canonical_json_sha256(protocol),
        },
        "verified_bindings": bindings,
        "data_roles": role_results,
        "legacy_negative_result_preserved": True,
        "target_contract": {
            "synthetic_truth_table_sha256": _synthetic_contract_sha256(),
            "step_support": list(range(9)),
            "transition_support": list(range(8)),
            "continuous_positive_magnitude_target_allowed": False,
        },
        "feature_contract": {
            "dimension": gp.FEATURE_DIMENSION,
            "layout": "82D_legacy_causal_plus_8D_sign_plus_7D_transition",
            "runtime_input_names": list(gp.RUNTIME_INPUT_NAMES),
            "forbidden_runtime_names": sorted(gp.FORBIDDEN_RUNTIME_NAMES),
            "teacher_or_other_candidate_visible": False,
        },
        "fold_contract": {
            "outer_folds": gp.OUTER_FOLD_COUNT,
            "inner_folds_per_outer": gp.INNER_FOLD_COUNT,
            "assignment_sha256": _fold_contract_sha256(),
            "all_calls_and_layers_stay_in_group": True,
        },
        "model_comparison": {
            "baseline": protocol["model_contract"]["count_baseline"]["family"],
            "primary": protocol["model_contract"]["primary_challenger"]["family"],
            "primary_fixed_before_labels": True,
            "post_label_switch_allowed": False,
        },
        "gate_contract": protocol["development_gates"],
        "tail_veto_contract": protocol["tail_veto_contract"],
        "access_ledger": {
            "protocol_json_opened": 1,
            "d0_result_json_opened": 1,
            "legacy_manifest_hashed": 1,
            "selection_json_opened": 3,
            "legacy_c355_result_json_opened": 1,
            "fresh_development_payload_opened": 0,
            "calibration_payload_opened": 0,
            "independent_test_payload_opened": 0,
            "c361_row_payload_opened": 0,
            "model_fits": 0,
            "gpu_operations": 0,
        },
        "claim_boundary": protocol["claim_boundary"],
        "process_boundary": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "forbidden_modules_loaded": forbidden_loaded,
            "stdlib_only": not forbidden_loaded,
        },
        "implementation_sha256": {
            "gripper_v2_protocol.py": gp.sha256_file(
                MODULE_DIRECTORY / "gripper_v2_protocol.py"
            ),
            "validate_v3_gripper_v2_protocol.py": gp.sha256_file(Path(__file__)),
        },
        "next_stage": protocol["next_stage"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo_root = args.repo_root.absolute()
        protocol_relative = _relative_path(args.protocol, context="protocol")
        if args.initialize_default_protocol:
            _write_new_json(repo_root / protocol_relative, gp.build_protocol_template())
        result = run(args)
        output = args.output
        if not output.is_absolute():
            output = repo_root / output
        _write_new_json(output, result)
    except (gp.GripperV2ProtocolError, OSError, ValueError) as error:
        print(f"V3-D1 protocol validation failed closed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "protocol": result["protocol"],
                "roles": result["data_roles"],
                "next_stage": result["next_stage"]["authorized"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
