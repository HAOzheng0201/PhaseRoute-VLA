from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = REPO_ROOT / "configs/route_first_active_pilot_protocol.json"
EXPECTED_PROTOCOL_SHA256 = (
    "fcb1c2a1fdf7ea3f79343f72d25240449500a5eac3fad1372f0808023888db4d"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_active_pilot_protocol_is_frozen_before_states_are_opened() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert _sha256(PROTOCOL) == EXPECTED_PROTOCOL_SHA256
    assert protocol["status"] == "PREREGISTERED_NOT_OPENED"
    assert protocol["schedule"]["engineering_smoke"]["episode_indices"] == [12]
    assert protocol["schedule"]["paired_pilot"]["episode_indices"] == [13]
    assert protocol["access_ledger"]["state12_smoke_opened"] is False
    assert protocol["access_ledger"]["state13_pilot_opened"] is False
    assert protocol["access_ledger"]["active_control_executed_for_this_stage"] is False
    assert (
        protocol["access_ledger"][
            "historical_D9_states40_to49_opened_for_this_stage"
        ]
        is False
    )


def test_active_pilot_binds_stage8_and_forbids_threshold_movement() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frozen = protocol["frozen_implementation"]

    assert frozen["stage8_commit"] == "00e832713b25b4a01d5a443d015613e18ae89fb4"
    assert _sha256(REPO_ROOT / frozen["stage8_verification_path"]) == frozen[
        "stage8_verification_sha256"
    ]
    assert _sha256(REPO_ROOT / frozen["runtime_path"]) == frozen["runtime_sha256"]
    assert _sha256(REPO_ROOT / frozen["controller_path"]) == frozen[
        "controller_sha256"
    ]
    assert protocol["shared_settings"]["threshold_movement"] is False
    assert protocol["shared_settings"]["router_refit"] is False
    assert protocol["failure_policy"][
        "do_not_move_threshold_after_opening_state12_or_state13"
    ] is True
