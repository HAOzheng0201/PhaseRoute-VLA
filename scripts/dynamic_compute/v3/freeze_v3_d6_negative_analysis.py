#!/usr/bin/env python3
"""Freeze V3-D6 negative-analysis attestation without tensor access."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path("reports/v3_d6_negative_analysis/result.json")
ANALYSIS_SHA256 = (
    "f9d36526615aff7c12e591076f5885c950b2bc4db5fde01595a1b579fe9f4726"
)
FORMAL = Path("results/v3/v3_d6_formal_development_result.json")
FORMAL_SHA256 = (
    "c8bda5b40afb93c5fe815e71224da1e0f99570e4b73970e4cf8489b78fd62fc6"
)
OUTPUT = Path("results/v3/v3_d6_negative_analysis.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D6 analysis metadata must be an object")
    return dict(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D6 analysis freeze requires clean worktree")
    analysis_path = REPO_ROOT / ANALYSIS
    formal_path = REPO_ROOT / FORMAL
    if sha256(analysis_path) != ANALYSIS_SHA256 or sha256(formal_path) != FORMAL_SHA256:
        raise PermissionError("V3-D6 analysis/formal SHA differs")
    analysis = json_object(analysis_path)
    formal = json_object(formal_path)
    if (
        analysis.get("status") != "PASS_V3_D6_NEGATIVE_RESULT_ANALYSIS"
        or analysis.get("formal_negative_result_reproduced") is not True
        or formal.get("status") != "NEGATIVE_V3_D6_DEVELOPMENT_SELECTION"
        or analysis.get("authorization", {}).get("next_stage")
        != "D7_PROTOCOL_DESIGN_ONLY_USING_D5_D6_AS_REUSED_DEVELOPMENT_EVIDENCE"
        or analysis.get("jackknife_effectiveness", {}).get(
            "folds_where_fifth_smallest_changed_pre_shrink_base"
        )
        != 0
    ):
        raise PermissionError("V3-D6 analysis semantics differ")
    result = {
        "schema_version": "phase-route-vla.v3.d6-negative-analysis-attestation.v1",
        "status": analysis["status"],
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": analysis["source_git_commit"],
        "freeze_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "formal_negative_result_reproduced": True,
        "formal_selection": analysis["formal_selection"],
        "error_summary": analysis["error_summary"],
        "false_safe_records_with_D5_comparison": analysis[
            "false_safe_records_with_D5_comparison"
        ],
        "route_change_from_locked_D5": analysis["route_change_from_locked_D5"],
        "jackknife_effectiveness": analysis["jackknife_effectiveness"],
        "threshold_stability": analysis["threshold_stability"],
        "posthoc_diagnostic_base_multipliers": analysis[
            "posthoc_diagnostic_base_multipliers"
        ],
        "scientific_interpretation": analysis["scientific_interpretation"],
        "D7_design_requirements": analysis["D7_design_requirements"],
        "artifacts": {
            "analysis_path": ANALYSIS.as_posix(),
            "analysis_sha256": ANALYSIS_SHA256,
            "formal_result_path": FORMAL.as_posix(),
            "formal_result_sha256": FORMAL_SHA256,
        },
        "access_ledger": analysis["access_ledger"],
        "authorization": analysis["authorization"],
        "claim_boundary": analysis["claim_boundary"],
    }
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("V3-D6 refuses to overwrite analysis attestation")
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.move(str(incomplete), str(output))
    digest = sha256(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
