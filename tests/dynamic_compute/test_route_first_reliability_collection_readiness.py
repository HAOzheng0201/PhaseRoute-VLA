from __future__ import annotations

import json
from pathlib import Path

import pytest

from a1.vla.dynamic_compute import route_first_reliability_collection as collection
from a1.vla.dynamic_compute.route_first_reliability import STAGE11D_PROTOCOL_SHA256
from a1.vla.dynamic_compute.route_first_reliability_artifacts import sha256_file
from a1.vla.dynamic_compute.route_first_reliability_state_binding import (
    STAGE11D_STATE_BINDING_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _readiness(runner_sha256: str) -> dict[str, object]:
    return {
        "schema_version": collection.STAGE11D_COLLECTION_READINESS_SCHEMA,
        "status": collection.STAGE11D_COLLECTION_READINESS_STATUS,
        "protocol_sha256": STAGE11D_PROTOCOL_SHA256,
        "state_binding_sha256": STAGE11D_STATE_BINDING_SHA256,
        "runner_files": {"runner.py": runner_sha256},
        "schedule": {"development_clusters": 120},
        "access_boundary": {
            "development_train": True,
            "calibration": False,
            "shadow_confirmation": False,
        },
        "authorization": {
            "original_A1_observation_collection": True,
            "same_noise_replay": False,
            "training": False,
            "active_control": False,
        },
    }


def test_production_collection_readiness_is_exact() -> None:
    readiness = collection.validate_collection_readiness(REPO_ROOT)
    assert readiness["schedule"]["development_clusters"] == 120
    assert readiness["access_boundary"]["calibration"] is False
    assert readiness["access_boundary"]["shadow_confirmation"] is False
    assert readiness["execution"]["GPU_queried_or_initialized"] is False
    assert readiness["execution"]["development_collection_started"] is False


def test_readiness_rejects_runner_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("print('frozen')\n", encoding="utf-8")
    readiness_path = tmp_path / collection.STAGE11D_COLLECTION_READINESS_RELATIVE_PATH
    readiness_path.parent.mkdir(parents=True)
    readiness_path.write_text(
        json.dumps(_readiness(sha256_file(runner))), encoding="utf-8"
    )
    monkeypatch.setattr(
        collection, "validate_checkpoint_inventory", lambda _root, _value: {}
    )
    assert collection.validate_collection_readiness(tmp_path)["status"].startswith(
        "PASS_"
    )
    runner.write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="runner file hash"):
        collection.validate_collection_readiness(tmp_path)


def test_readiness_rejects_withheld_split_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("print('frozen')\n", encoding="utf-8")
    value = _readiness(sha256_file(runner))
    value["access_boundary"]["calibration"] = True
    readiness_path = tmp_path / collection.STAGE11D_COLLECTION_READINESS_RELATIVE_PATH
    readiness_path.parent.mkdir(parents=True)
    readiness_path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        collection, "validate_checkpoint_inventory", lambda _root, _value: {}
    )
    with pytest.raises(PermissionError, match="readiness semantics"):
        collection.validate_collection_readiness(tmp_path)
