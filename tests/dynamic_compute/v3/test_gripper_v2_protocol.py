from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIRECTORY = REPO_ROOT / "a1" / "vla" / "dynamic_compute" / "v3"
LEGACY_EVIDENCE_ROOT = (
    REPO_ROOT / "artifacts" / "phase_route_v3" / "legacy_source"
)
sys.path.insert(0, str(MODULE_DIRECTORY))
import gripper_v2_protocol as gp  # noqa: E402


def _selection(role: str, episodes: range) -> dict[str, object]:
    return {
        "schema_version": "phase-route-vla.v3.data-lineage-selection.v1",
        "suite": "libero_10",
        "role": role,
        "records": [
            {
                "task_id": task,
                "episode_index": episode,
                "seed": 20260820 + task * 100 + episode,
            }
            for task in range(10)
            for episode in episodes
        ],
    }


def test_frozen_template_has_expected_lineage_roles_and_claim_boundary() -> None:
    protocol = gp.build_protocol_template()
    assert protocol["schema_version"] == gp.PROTOCOL_SCHEMA_VERSION
    assert protocol["protocol_id"] == gp.PROTOCOL_ID
    assert protocol["status"] == gp.PROTOCOL_STATUS
    assert protocol["lineage"]["d0_result_sha256"] == (
        "64d1159b3941fe1e7b806da981a0f47297758dcc2cad87d4e283d03db3a71c4b"
    )
    assert protocol["lineage"]["legacy_gripper_disposition"] == (
        "BASELINE_OR_FAIL_NEGATIVE_RESULT_FROZEN"
    )
    assert [item["key_count"] for item in protocol["data_roles"]] == [180, 100, 100]
    assert not any(protocol["claim_boundary"].values())
    assert protocol["next_stage"]["calibration_or_test_authorized"] is False
    assert protocol["next_stage"]["runtime_or_control_authorized"] is False


def test_template_copy_canonical_hash_and_exact_mutation_rejection(tmp_path: Path) -> None:
    first = gp.build_protocol_template()
    second = gp.build_protocol_template()
    first["scope"]["training_allowed"] = True
    assert second["scope"]["training_allowed"] is False
    assert gp.canonical_json_sha256(second) == gp.canonical_json_sha256(
        json.loads(json.dumps(second, sort_keys=False))
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(second) + "\n", encoding="utf-8")
    assert gp.load_protocol(protocol_path) == second
    mutated = gp.build_protocol_template()
    mutated["runtime_input_contract"]["feature_dimension"] = 98
    with pytest.raises(gp.ProtocolSchemaError, match="feature_dimension"):
        gp.validate_protocol_document(mutated)
    mutated = gp.build_protocol_template()
    mutated["claim_boundary"]["model_trained"] = True
    with pytest.raises(gp.ProtocolSchemaError, match="model_trained"):
        gp.validate_protocol_document(mutated)


def test_duplicate_nonfinite_and_non_object_json_fail_closed() -> None:
    with pytest.raises(gp.ProtocolSchemaError, match="duplicate"):
        gp.decode_json_bytes(b'{"a":1,"a":2}', context="duplicate")
    with pytest.raises(gp.ProtocolSchemaError, match="non-finite"):
        gp.decode_json_bytes(b'{"a":NaN}', context="nan")
    with pytest.raises(gp.ProtocolSchemaError, match="object"):
        gp.validate_protocol_document([])


def test_gripper_state_truth_table_and_invalid_values() -> None:
    assert gp.gripper_state(-100.0) == 0
    assert gp.gripper_state(-1e-12) == 0
    assert gp.gripper_state(0.0) == 1
    assert gp.gripper_state(1e-12) == 1
    assert gp.gripper_state(100) == 1
    for invalid in (True, False, float("nan"), float("inf"), "0"):
        with pytest.raises(gp.TargetConstructionError):
            gp.gripper_state(invalid)  # type: ignore[arg-type]


def test_identical_targets_are_exact_zero() -> None:
    teacher = [-1.0] * 4 + [1.0] * 4
    result = gp.construct_gripper_targets({11: teacher, 13: teacher}, teacher)
    assert result["teacher_state"] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert result["teacher_transition_pattern"] == [0, 0, 0, 1, 0, 0, 0]
    for layer in result["layers"]:
        assert layer["step_count"] == 0
        assert layer["transition_count"] == 0
        assert layer["step_occurrence"] is False
        assert layer["transition_occurrence"] is False
        assert layer["first_transition_mismatch"] == 0


def test_single_step_and_all_flip_preserve_discrete_semantics() -> None:
    teacher = [-1.0] * 8
    one_step = [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    all_flip = [1.0] * 8
    result = gp.construct_gripper_targets({11: one_step, 13: all_flip}, teacher)
    layer11, layer13 = result["layers"]
    assert layer11["step_mismatch_bits"] == [0, 0, 0, 1, 0, 0, 0, 0]
    assert layer11["step_count"] == 1
    assert layer11["transition_mismatch_bits"] == [0, 0, 1, 1, 0, 0, 0]
    assert layer11["transition_count"] == 2
    assert layer11["first_transition_mismatch"] == 3
    assert layer13["step_count"] == 8
    assert layer13["transition_count"] == 0
    assert layer13["step_occurrence"] is True
    assert layer13["transition_occurrence"] is False


def test_early_and_late_transitions_produce_timing_target() -> None:
    teacher = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    late = [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    early = [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    result = gp.construct_gripper_targets({11: late, 13: early}, teacher)
    layer11, layer13 = result["layers"]
    assert layer11["transition_mismatch_bits"] == [0, 0, 1, 0, 1, 0, 0]
    assert layer11["first_transition_mismatch"] == 3
    assert layer13["transition_mismatch_bits"] == [1, 0, 1, 0, 0, 0, 0]
    assert layer13["first_transition_mismatch"] == 1


def test_target_builder_rejects_geometry_layer_and_nonfinite_errors() -> None:
    valid = [-1.0] * 8
    with pytest.raises(gp.TargetConstructionError, match="exactly layers"):
        gp.construct_gripper_targets({11: valid}, valid)
    with pytest.raises(gp.TargetConstructionError, match="exactly 8"):
        gp.construct_gripper_targets({11: valid[:-1], 13: valid}, valid)
    invalid = list(valid)
    invalid[3] = float("nan")
    with pytest.raises(gp.TargetConstructionError, match="finite"):
        gp.construct_gripper_targets({11: invalid, 13: valid}, valid)
    with pytest.raises(gp.TargetConstructionError, match="0/1"):
        gp.transition_pattern([0, 0, 0, 0, 0, 0, 0, 2])


def test_runtime_allowlist_is_exact_and_rejects_leakage() -> None:
    assert gp.validate_runtime_input_names(gp.RUNTIME_INPUT_NAMES) == gp.RUNTIME_INPUT_NAMES
    with pytest.raises(gp.ProtocolSchemaError, match="order or membership"):
        gp.validate_runtime_input_names(gp.RUNTIME_INPUT_NAMES[:-1])
    with pytest.raises(gp.ProtocolSchemaError, match="leakage"):
        gp.validate_runtime_input_names(
            (*gp.RUNTIME_INPUT_NAMES, "layer27_candidate_action")
        )
    with pytest.raises(gp.ProtocolSchemaError):
        gp.validate_runtime_input_names("instruction_summary")


def test_eighteen_by_seventeen_loeo_is_exact_and_layer_invariant() -> None:
    assert gp.OUTER_FOLD_COUNT == 18
    assert gp.INNER_FOLD_COUNT == 17
    assert [gp.outer_validation_episode(fold) for fold in range(18)] == list(
        range(12, 30)
    )
    for outer in range(18):
        remaining = gp.inner_validation_episodes(outer)
        assert len(remaining) == 17
        assert gp.outer_validation_episode(outer) not in remaining
        assert [gp.inner_fold_id(outer, episode) for episode in remaining] == list(
            range(17)
        )
    for task in range(10):
        for episode in range(12, 30):
            layer11 = gp.grouped_fold_assignment(
                task_id=task, episode_index=episode, candidate_layer=11
            )
            layer13 = gp.grouped_fold_assignment(
                task_id=task, episode_index=episode, candidate_layer=13
            )
            assert layer11 == layer13
            assert layer11["outer_fold"] == episode - 12
            assert len(layer11["inner_fold_by_outer_train"]) == 17


def test_grouped_fold_invalid_values_fail_closed() -> None:
    with pytest.raises(gp.FoldContractError):
        gp.grouped_fold_assignment(task_id=10, episode_index=12, candidate_layer=11)
    with pytest.raises(gp.FoldContractError):
        gp.grouped_fold_assignment(task_id=0, episode_index=30, candidate_layer=11)
    with pytest.raises(gp.FoldContractError):
        gp.grouped_fold_assignment(task_id=0, episode_index=12, candidate_layer=27)
    with pytest.raises(gp.FoldContractError):
        gp.inner_fold_id(0, 12)


@pytest.mark.parametrize(
    ("role", "episodes", "count"),
    [
        ("development_v2", range(12, 30), 180),
        ("calibration_v2", range(30, 40), 100),
        ("independent_test_v2", range(40, 50), 100),
    ],
)
def test_role_selection_validation(role: str, episodes: range, count: int) -> None:
    selection = _selection(role, episodes)
    gp.validate_selection_document(
        selection, role=role, episodes=episodes, expected_count=count
    )
    selection["records"].pop()  # type: ignore[union-attr]
    with pytest.raises(gp.ProtocolSchemaError, match="frozen role grid"):
        gp.validate_selection_document(
            selection, role=role, episodes=episodes, expected_count=count
        )


def test_protocol_path_rejects_symlink_and_traversal(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "protocol.json").write_text(
        json.dumps(gp.build_protocol_template()), encoding="utf-8"
    )
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(gp.ProtocolPathError, match="symlink"):
        gp.load_protocol(alias / "protocol.json")
    with pytest.raises(gp.ProtocolPathError, match="contained"):
        gp.resolve_regular_file(real, "../escape.json")


def test_development_gates_are_complete_and_sign_gate_is_exact() -> None:
    gates = gp.build_protocol_template()["development_gates"]
    assert gates["occurrence"]["all_target_scopes_required"] is True
    assert gates["conditional_count"]["minimum_strictly_improved_layer_target_scopes"] == 3
    assert gates["conditional_count"]["worst_layer_target_ratio_at_most"] == 1.01
    assert gates["expected_fraction"]["all_target_scopes_required"] is True
    assert gates["group_robustness"]["minimum_improved_outer_episodes"] == 13
    assert gates["group_robustness"]["metric"] == (
        "outer_episode_conditional_count_nll_improvement"
    )
    assert gates["group_robustness"]["missing_positive_task_layer_target_cell"] == (
        "INCONCLUSIVE"
    )
    p_value = sum(math.comb(18, k) for k in range(13, 19)) / (2**18)
    assert p_value < 0.05
    assert sum(math.comb(18, k) for k in range(12, 19)) / (2**18) >= 0.05


def test_model_and_tail_contract_cannot_be_compensated_or_switched() -> None:
    protocol = gp.build_protocol_template()
    model = protocol["model_contract"]
    assert model["count_baseline"]["family"] == "zero_truncated_binomial_glm"
    assert model["primary_challenger"]["family"] == "ordinal_cumulative_link_glm"
    assert model["primary_challenger"]["trainable_cutpoints"] is True
    assert (
        model["primary_challenger"][
            "cutpoints_count_total_across_layers_and_targets"
        ]
        == 26
    )
    assert model["shared_constraints"]["linear_feature_head_bias"] is False
    assert model["shared_constraints"][
        "ordinal_cutpoints_are_not_linear_feature_bias"
    ] is True
    assert model["primary_family_fixed_before_labels"] == (
        "ordinal_cumulative_link_glm"
    )
    assert model["post_label_family_switch_allowed"] is False
    tail = protocol["tail_veto_contract"]
    assert tail["independent_head"] is True
    assert tail["gripper_score_may_compensate_tail_failure"] is False
    assert tail["missing_or_nonfinite_tail"] == "force_deeper_compute"
    calibration = protocol["future_calibration_contract"]
    assert calibration["ucb_method"] == (
        "one_sided_exact_clopper_pearson_on_cluster_events"
    )
    assert calibration["cluster_denominator"] == (
        "clusters_with_at_least_one_predicted_safe_call"
    )
    assert calibration["minimum_safe_coverage"] == 0.10
    assert protocol["runtime_input_contract"]["context_shapes"][
        "candidate_layer"
    ] == []


def test_module_is_standard_library_only_and_contains_no_trainer() -> None:
    source = (MODULE_DIRECTORY / "gripper_v2_protocol.py").read_text(encoding="utf-8")
    for forbidden in (
        "import torch",
        "from torch",
        "import numpy",
        "from numpy",
        "pickle.load(",
        "torch.load(",
        "numpy.load(",
    ):
        assert forbidden not in source
    assert "def train" not in source


def test_cli_relocated_validation_is_stdlib_only_and_fail_closed(tmp_path: Path) -> None:
    relocated_repo = tmp_path / "repo"
    relocated_legacy = tmp_path / "legacy_source"
    repo_files = [
        "results/v3/v3_d0_data_lineage_audit.json",
        "docs/research/v3/legacy_evidence_manifest.json",
        "configs/research/v3/data_lineage/development_v2.json",
        "configs/research/v3/data_lineage/calibration_v2.json",
        "configs/research/v3/data_lineage/independent_test_v2.json",
    ]
    for relative in repo_files:
        destination = relocated_repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    legacy_relative = (
        "reports/phase_route_v2_stage_c355_development_predictor_training_"
        "20260818_v1/result.json"
    )
    legacy_source = LEGACY_EVIDENCE_ROOT / legacy_relative
    legacy_destination = relocated_legacy / legacy_relative
    legacy_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(legacy_source, legacy_destination)

    script = REPO_ROOT / "scripts/dynamic_compute/v3/validate_v3_gripper_v2_protocol.py"
    output = relocated_repo / "results/v3/validation.json"
    command = [
        sys.executable,
        str(script),
        "--repo-root",
        str(relocated_repo),
        "--legacy-source-root",
        str(relocated_legacy),
        "--protocol",
        "configs/research/v3/gripper_v2/protocol.json",
        "--initialize-default-protocol",
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "7"
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "PASS_D1_GRIPPER_V2_PROTOCOL_FROZEN"
    assert result["process_boundary"] == {
        "cuda_visible_devices": "",
        "forbidden_modules_loaded": [],
        "stdlib_only": True,
    }
    assert result["access_ledger"]["fresh_development_payload_opened"] == 0
    assert result["access_ledger"]["c361_row_payload_opened"] == 0
    assert result["fold_contract"]["outer_folds"] == 18
    assert result["feature_contract"]["dimension"] == 97

    repeated = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode == 2
    assert "refusing to overwrite" in repeated.stderr

    protocol_path = relocated_repo / "configs/research/v3/gripper_v2/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["model_contract"]["primary_family_fixed_before_labels"] = (
        "post_hoc_model"
    )
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    mutation_command = [item for item in command if item != "--initialize-default-protocol"]
    mutation_command[-1] = str(relocated_repo / "results/v3/mutated.json")
    mutated = subprocess.run(
        mutation_command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert mutated.returncode == 2
    assert "primary_family_fixed_before_labels" in mutated.stderr
