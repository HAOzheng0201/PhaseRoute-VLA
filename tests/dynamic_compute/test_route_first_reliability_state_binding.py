from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from a1.vla.dynamic_compute import route_first_reliability_state_binding as binding
from a1.vla.dynamic_compute.route_first_reliability_artifacts import (
    STAGE11D_RUNNER_READINESS_RELATIVE_PATH,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _copy(relative: Path, destination: Path) -> None:
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / relative, target)


def _copy_tracked_evidence(destination: Path) -> None:
    for relative in (
        binding.STAGE11D_STATE_BINDING_RELATIVE_PATH,
        binding.STAGE11D_STATE_RESULT_RELATIVE_PATH,
        STAGE11D_RUNNER_READINESS_RELATIVE_PATH,
    ):
        _copy(relative, destination)


def test_stage11d_tracked_state_binding_is_exact() -> None:
    value = binding.load_stage11d_state_binding(REPO_ROOT)
    assert value["local_state_payload"]["records"] == 200
    assert value["local_state_payload"]["bytes"] == 197546
    assert value["authorization"]["original_A1_collection_started"] is False
    assert value["authorization"]["active_control_started"] is False


def test_stage11d_tracked_result_hash_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_tracked_evidence(tmp_path)
    monkeypatch.setattr(binding, "validate_stage11d_protocol", lambda _root: {})
    monkeypatch.setattr(binding, "validate_state_runner_readiness", lambda _root: {})
    result_path = tmp_path / binding.STAGE11D_STATE_RESULT_RELATIVE_PATH
    result_path.write_text(
        result_path.read_text(encoding="utf-8").replace(
            '"scheduled_clusters": 200', '"scheduled_clusters": 201'
        ),
        encoding="utf-8",
    )
    with pytest.raises(binding.Stage11DStateBindingError, match="result SHA-256"):
        binding.load_stage11d_state_binding(tmp_path)


def test_stage11d_local_payload_hash_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_tracked_evidence(tmp_path)
    monkeypatch.setattr(binding, "validate_stage11d_protocol", lambda _root: {})
    monkeypatch.setattr(binding, "validate_state_runner_readiness", lambda _root: {})
    for relative in (
        Path("runs/route_first_stage11d_states/state_attestation.json"),
        Path("runs/route_first_stage11d_states/fresh_states.pt"),
    ):
        _copy(relative, tmp_path)
    payload = tmp_path / "runs/route_first_stage11d_states/fresh_states.pt"
    with payload.open("ab") as output:
        output.write(b"tampered")
    with pytest.raises(binding.Stage11DStateBindingError, match="SHA or size"):
        binding.validate_local_stage11d_state_artifacts(tmp_path)


@pytest.mark.skipif(
    not (REPO_ROOT / "runs/route_first_stage11d_states/fresh_states.pt").exists(),
    reason="ignored Stage-11D payload is not present in a clean source checkout",
)
def test_stage11d_production_local_artifacts_and_payload_are_bound() -> None:
    evidence = binding.validate_local_stage11d_state_artifacts(REPO_ROOT)
    assert evidence["attestation"]["audit"]["byte_identical_records"] == 200
    schedule, states, attestation = binding.load_bound_stage11d_states(REPO_ROOT)
    assert len(schedule) == 200
    assert all(len(states[task]) == 20 for task in range(10))
    assert attestation["payload_sha256"] == (
        "2de72279a8dc60f7853ad698b2d710e6a73c83a625b26ed70e74e0d7d76856db"
    )
