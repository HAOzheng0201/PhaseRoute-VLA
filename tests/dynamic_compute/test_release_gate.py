import json
from pathlib import Path

from a1.vla.dynamic_compute import release


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return release.sha256_file(path)


def test_release_gate_accepts_frozen_equivalent_result(tmp_path, monkeypatch):
    checkpoint = tmp_path / release.CHECKPOINT_RELATIVE_PATH
    config = tmp_path / release.CONFIG_RELATIVE_PATH
    dataset_statistics = tmp_path / release.DATASET_STATISTICS_RELATIVE_PATH
    threshold = tmp_path / release.THRESHOLD_RELATIVE_PATH
    paired = tmp_path / release.PAIRED_RESULT_RELATIVE_PATH
    checkpoint_sha = _write(checkpoint, b"checkpoint")
    config_sha = _write(config, b"config")
    dataset_statistics_sha = _write(dataset_statistics, b"statistics")
    threshold_sha = _write(threshold, b"threshold")
    result = {
        "status": "PASS",
        "scope": "m420b_rp_pep_paired_closed_loop_summary",
        "paired_episodes": 20,
        "total_rollouts": 40,
        "equivalence": {
            "success_mismatches": 0,
            "action_chunk_sha256_mismatches": 0,
            "exit_layer_sequence_mismatches": 0,
            "policy_call_count_mismatches": 0,
        },
        "fm_solver_calls": {"reduction_fraction": 0.41},
        "policy_latency": {
            "weighted_mean_reduction_fraction": 0.31,
            "median_reduction_fraction": 0.29,
        },
        "gates": {"trajectory_equivalence": True, "latency": True},
        "baseline_successes": 20,
        "rp_pep_successes": 20,
    }
    paired.parent.mkdir(parents=True, exist_ok=True)
    paired.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(release, "CHECKPOINT_SHA256", checkpoint_sha)
    monkeypatch.setattr(release, "CONFIG_SHA256", config_sha)
    monkeypatch.setattr(
        release, "DATASET_STATISTICS_SHA256", dataset_statistics_sha
    )
    monkeypatch.setattr(release, "THRESHOLD_SHA256", threshold_sha)
    monkeypatch.setattr(release, "PAIRED_RESULT_SHA256", release.sha256_file(paired))

    output = release.validate_rp_pep_release(tmp_path)

    assert output["status"] == "PASS"
    assert output["release_method"] == "rp_pep"
    assert output["runtime_default_enabled"] is False
    assert output["learned_router_runtime_allowed"] is False
    assert all(output["checks"].values())


def test_release_gate_rejects_failed_science_gate(tmp_path, monkeypatch):
    checkpoint = tmp_path / release.CHECKPOINT_RELATIVE_PATH
    config = tmp_path / release.CONFIG_RELATIVE_PATH
    dataset_statistics = tmp_path / release.DATASET_STATISTICS_RELATIVE_PATH
    threshold = tmp_path / release.THRESHOLD_RELATIVE_PATH
    paired = tmp_path / release.PAIRED_RESULT_RELATIVE_PATH
    monkeypatch.setattr(release, "CHECKPOINT_SHA256", _write(checkpoint, b"c"))
    monkeypatch.setattr(release, "CONFIG_SHA256", _write(config, b"cfg"))
    monkeypatch.setattr(
        release,
        "DATASET_STATISTICS_SHA256",
        _write(dataset_statistics, b"stats"),
    )
    monkeypatch.setattr(release, "THRESHOLD_SHA256", _write(threshold, b"t"))
    paired.parent.mkdir(parents=True, exist_ok=True)
    paired.write_text(
        json.dumps(
            {
                "status": "PASS",
                "scope": "m420b_rp_pep_paired_closed_loop_summary",
                "paired_episodes": 20,
                "total_rollouts": 40,
                "equivalence": {
                    "success_mismatches": 0,
                    "action_chunk_sha256_mismatches": 0,
                    "exit_layer_sequence_mismatches": 0,
                    "policy_call_count_mismatches": 0,
                },
                "fm_solver_calls": {"reduction_fraction": 0.41},
                "policy_latency": {
                    "weighted_mean_reduction_fraction": 0.31,
                    "median_reduction_fraction": 0.29,
                },
                "gates": {"trajectory_equivalence": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "PAIRED_RESULT_SHA256", release.sha256_file(paired))

    output = release.validate_rp_pep_release(tmp_path)

    assert output["status"] == "FAIL"
    assert output["checks"]["all_frozen_gates"] is False
