from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HOME", "/tmp/phase-route-vla-test-hf")
os.environ.setdefault("DATA_DIR", str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("VLA_CONFIG_YAML", "libero_simulation.yaml")

import robot_experiments.libero.stage11_vla_utils as stage11_utils


def test_measurement_start_failure_does_not_change_returned_action(monkeypatch) -> None:
    sentinel = object()

    def failed_probe(_model):
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(stage11_utils, "_probe_for", failed_probe)
    monkeypatch.setattr(
        stage11_utils,
        "get_stage1_vla_action",
        lambda *args, **kwargs: sentinel,
    )
    result = stage11_utils.get_vla_action(
        SimpleNamespace(),
        object(),
        "cpu",
        {},
        "task",
    )
    assert result is sentinel


def test_measurement_write_failure_does_not_change_returned_action(monkeypatch) -> None:
    sentinel = object()

    class Probe:
        def start_call(self, _context):
            return None

        def finish_call(self, **_kwargs):
            return {"record": True}

    monkeypatch.setattr(stage11_utils, "_probe_for", lambda _model: Probe())
    monkeypatch.setattr(
        stage11_utils,
        "get_stage1_vla_action",
        lambda *args, **kwargs: sentinel,
    )
    monkeypatch.setattr(
        stage11_utils,
        "_append_record",
        lambda _record: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    result = stage11_utils.get_vla_action(
        SimpleNamespace(),
        object(),
        "cpu",
        {},
        "task",
    )
    assert result is sentinel
