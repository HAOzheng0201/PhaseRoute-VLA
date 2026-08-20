#!/usr/bin/env python3
"""Freeze the V3-D5 negative-analysis attestation without tensor access."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path("reports/v3_d5_negative_analysis/result.json")
ANALYSIS_SHA256 = (
    "e4705fcbaa0e1a917df2a928ac1afc62c4921757ede682f8e6ca8c8df2aee9b4"
)
FORMAL = Path("results/v3/v3_d5_formal_development_result.json")
FORMAL_SHA256 = (
    "f08e35e9588f44900d6e714dc45c7afb9e1cc7586e8bbbfade488f3ed783b6f8"
)
OUTPUT = Path("results/v3/v3_d5_negative_analysis.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D5 analysis metadata must be an object")
    return dict(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D5 analysis freeze requires clean worktree")
    analysis_path = REPO_ROOT / ANALYSIS
    formal_path = REPO_ROOT / FORMAL
    if sha256(analysis_path) != ANALYSIS_SHA256 or sha256(formal_path) != FORMAL_SHA256:
        raise PermissionError("V3-D5 analysis/formal SHA differs")
    analysis = json_object(analysis_path)
    formal = json_object(formal_path)
    if (
        analysis.get("status") != "PASS_V3_D5_NEGATIVE_RESULT_ANALYSIS"
        or analysis.get("formal_negative_result_reproduced") is not True
        or formal.get("status") != "NEGATIVE_V3_D5_DEVELOPMENT_GATE"
        or analysis.get("authorization", {}).get("next_stage")
        != "D6_PROTOCOL_DESIGN_ONLY_USING_D5_AS_DEVELOPMENT_EVIDENCE"
    ):
        raise PermissionError("V3-D5 analysis semantics differ")
    result = {
        "schema_version": "phase-route-vla.v3.d5-negative-analysis-attestation.v1",
        "status": analysis["status"],
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "source_git_commit": analysis["source_git_commit"],
        "freeze_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "formal_negative_result_reproduced": True,
        "formal_gate": analysis["formal_gate"],
        "error_summary": analysis["error_summary"],
        "false_safe_records": analysis["false_safe_records"],
        "threshold_stability": analysis["threshold_stability"],
        "posthoc_diagnostic_threshold_multipliers": analysis[
            "posthoc_diagnostic_threshold_multipliers"
        ],
        "scientific_interpretation": analysis["scientific_interpretation"],
        "artifacts": {
            "analysis_path": ANALYSIS.as_posix(),
            "analysis_sha256": ANALYSIS_SHA256,
            "formal_result_path": FORMAL.as_posix(),
            "formal_result_sha256": FORMAL_SHA256,
        },
        "authorization": analysis["authorization"],
        "claim_boundary": analysis["claim_boundary"],
    }
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("V3-D5 refuses to overwrite analysis attestation")
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
