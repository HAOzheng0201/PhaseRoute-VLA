#!/usr/bin/env python3
"""Export deterministic paper assets from the frozen V3-D9 formal result."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT = Path("results/v3/v3_d9_final_result.json")
INPUT_SHA256 = "4df77237e84ad82b05ae67145e52000b0e3430f34b6f69fcbee743687ac11952"
OUTPUT_DIR = Path("docs/research/v3/paper_assets")
ATTESTATION = Path("results/v3/v3_d10_paper_analysis.json")
STATUS = "PASS_V3_D10_FROZEN_PAPER_ASSET_EXPORT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError("D10 input must be a JSON object")
    return dict(value)


def write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _svg(result: Mapping[str, Any]) -> str:
    success = result["success"]
    efficiency = result["efficiency"]
    a1_success = 100.0 * success["A1_success_rate"]
    phase_success = 100.0 * success["PhaseRoute_success_rate"]
    a1_fm = efficiency["A1_FM_calls_per_policy_call"]
    phase_fm = efficiency["PhaseRoute_FM_calls_per_policy_call"]
    route_total = efficiency["PhaseRoute_policy_calls"]
    route = {
        "L11": 100.0 * efficiency["L11_calls"] / route_total,
        "L13": 100.0 * efficiency["L13_calls"] / route_total,
        "L27": 100.0 * efficiency["L27_calls"] / route_total,
    }
    a1_success_h = 2.8 * a1_success
    phase_success_h = 2.8 * phase_success
    a1_fm_h = 280.0 * a1_fm / max(a1_fm, phase_fm)
    phase_fm_h = 280.0 * phase_fm / max(a1_fm, phase_fm)
    # The right panel reserves exactly 320 px for the 100% stacked route bar.
    l11_w = 3.2 * route["L11"]
    l13_w = 3.2 * route["L13"]
    l27_w = 3.2 * route["L27"]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" viewBox="0 0 1200 650" role="img" aria-labelledby="title desc">
  <title id="title">PhaseRoute-VLA V3 D9 independent test summary</title>
  <desc id="desc">Success, flow-matching compute, route distribution, and same-noise safety results.</desc>
  <rect width="1200" height="650" fill="#fbfcfe"/>
  <text x="60" y="60" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#17233c">PhaseRoute-VLA · Paired Active Independent Test</text>
  <text x="60" y="92" font-family="Arial, sans-serif" font-size="16" fill="#5b6578">LIBERO-10 · 100 paired episodes · frozen A1 checkpoint · 18/18 primary gates PASS</text>

  <rect x="50" y="125" width="335" height="440" rx="16" fill="#ffffff" stroke="#dce3ed"/>
  <text x="75" y="165" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="#17233c">Task success ↑</text>
  <line x1="95" y1="475" x2="345" y2="475" stroke="#aeb8c7"/>
  <rect x="120" y="{475-a1_success_h:.2f}" width="75" height="{a1_success_h:.2f}" rx="5" fill="#7c8da6"/>
  <rect x="245" y="{475-phase_success_h:.2f}" width="75" height="{phase_success_h:.2f}" rx="5" fill="#2878d0"/>
  <text x="157.5" y="{462-a1_success_h:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#46546a">{a1_success:.0f}%</text>
  <text x="282.5" y="{462-phase_success_h:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#185b9f">{phase_success:.0f}%</text>
  <text x="157.5" y="505" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#46546a">A1</text>
  <text x="282.5" y="505" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#185b9f">PhaseRoute</text>
  <text x="217.5" y="542" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#16815d">+3 percentage points</text>

  <rect x="415" y="125" width="335" height="440" rx="16" fill="#ffffff" stroke="#dce3ed"/>
  <text x="440" y="165" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="#17233c">FM calls / policy call ↓</text>
  <line x1="460" y1="475" x2="710" y2="475" stroke="#aeb8c7"/>
  <rect x="485" y="{475-a1_fm_h:.2f}" width="75" height="{a1_fm_h:.2f}" rx="5" fill="#7c8da6"/>
  <rect x="610" y="{475-phase_fm_h:.2f}" width="75" height="{phase_fm_h:.2f}" rx="5" fill="#f29a2e"/>
  <text x="522.5" y="{462-a1_fm_h:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#46546a">{a1_fm:.2f}</text>
  <text x="647.5" y="{462-phase_fm_h:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#a85b00">{phase_fm:.2f}</text>
  <text x="522.5" y="505" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#46546a">A1</text>
  <text x="647.5" y="505" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#a85b00">PhaseRoute</text>
  <text x="582.5" y="542" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#16815d">−{100*efficiency['measured_FM_calls_per_policy_call_reduction']:.2f}% normalized compute</text>

  <rect x="780" y="125" width="370" height="440" rx="16" fill="#ffffff" stroke="#dce3ed"/>
  <text x="805" y="165" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="#17233c">PhaseRoute decisions</text>
  <text x="805" y="205" font-family="Arial, sans-serif" font-size="16" fill="#5b6578">Early exit: {100*efficiency['PhaseRoute_early_exit_call_fraction']:.2f}%</text>
  <rect x="805" y="230" width="{l11_w:.2f}" height="46" rx="6" fill="#24a47a"/>
  <rect x="{805+l11_w:.2f}" y="230" width="{l13_w:.2f}" height="46" fill="#4f8edc"/>
  <rect x="{805+l11_w+l13_w:.2f}" y="230" width="{l27_w:.2f}" height="46" rx="6" fill="#b9c2cf"/>
  <circle cx="820" cy="316" r="7" fill="#24a47a"/><text x="838" y="322" font-family="Arial, sans-serif" font-size="16" fill="#354158">L11 · {route['L11']:.2f}%</text>
  <circle cx="820" cy="354" r="7" fill="#4f8edc"/><text x="838" y="360" font-family="Arial, sans-serif" font-size="16" fill="#354158">L13 · {route['L13']:.2f}%</text>
  <circle cx="820" cy="392" r="7" fill="#b9c2cf"/><text x="838" y="398" font-family="Arial, sans-serif" font-size="16" fill="#354158">L27 fallback · {route['L27']:.2f}%</text>
  <line x1="805" y1="425" x2="1125" y2="425" stroke="#dce3ed"/>
  <text x="805" y="462" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#17233c">Same-noise safety</text>
  <text x="805" y="494" font-family="Arial, sans-serif" font-size="16" fill="#16815d">0 unsafe early calls</text>
  <text x="805" y="522" font-family="Arial, sans-serif" font-size="16" fill="#5b6578">Exact CP-UCB95: {100*result['safety']['false_safe_cluster_exact_CP_UCB95']:.3f}%</text>

  <text x="60" y="616" font-family="Arial, sans-serif" font-size="14" fill="#687386">FM metric is trajectory-length normalized. L27 is a same-noise consistency teacher, not an expert or task-success certificate.</text>
</svg>
'''


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("D10 paper export is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D10 paper export requires a clean source commit")
    input_path = REPO_ROOT / INPUT
    if sha256_file(input_path) != INPUT_SHA256:
        raise PermissionError("D10 frozen D9 input SHA-256 differs")
    result = json_object(input_path)
    if (
        result.get("status") != "PASS_V3_D9_PAIRED_ACTIVE_INDEPENDENT_TEST"
        or result.get("all_primary_gates_pass") is not True
        or len(result.get("gate_checks", {})) != 18
        or not all(result["gate_checks"].values())
        or result.get("claim_boundary", {}).get(
            "early_exit_and_failure_cooccurrence_proves_causation"
        )
        is not False
    ):
        raise PermissionError("D10 frozen D9 result semantics differ")

    output = REPO_ROOT / OUTPUT_DIR
    attestation = REPO_ROOT / ATTESTATION
    if output.exists() or attestation.exists() or attestation.with_suffix(".sha256").exists():
        raise FileExistsError("D10 paper asset export refuses overwrite")
    output.mkdir(parents=True, exist_ok=False)
    success = result["success"]
    efficiency = result["efficiency"]
    safety = result["safety"]
    write_csv(
        output / "d9_main_results.csv",
        ("metric", "A1", "PhaseRoute", "difference_or_reduction", "role"),
        [
            (
                "success_rate",
                success["A1_success_rate"],
                success["PhaseRoute_success_rate"],
                success["PhaseRoute_minus_A1_success_rate"],
                "primary",
            ),
            (
                "FM_calls_per_policy_call",
                efficiency["A1_FM_calls_per_policy_call"],
                efficiency["PhaseRoute_FM_calls_per_policy_call"],
                efficiency["measured_FM_calls_per_policy_call_reduction"],
                "primary",
            ),
            (
                "early_exit_call_fraction",
                "NA",
                efficiency["PhaseRoute_early_exit_call_fraction"],
                "NA",
                "primary",
            ),
            (
                "false_safe_cluster_CP_UCB95",
                "NA",
                safety["false_safe_cluster_exact_CP_UCB95"],
                "NA",
                "primary",
            ),
        ],
    )
    per_task_rows = []
    for task in range(10):
        value = result["per_task"][str(task)]
        per_task_rows.append(
            (
                task,
                value["A1_successes"],
                value["PhaseRoute_successes"],
                value["PhaseRoute_minus_A1_successes"],
                value["A1_FM_calls_per_policy_call"],
                value["PhaseRoute_FM_calls_per_policy_call"],
                value["FM_calls_per_policy_call_reduction"],
                value["early_exit_calls"],
            )
        )
    write_csv(
        output / "d9_per_task_results.csv",
        (
            "task_id",
            "A1_successes",
            "PhaseRoute_successes",
            "success_difference",
            "A1_FM_per_call",
            "PhaseRoute_FM_per_call",
            "FM_reduction",
            "PhaseRoute_early_exit_calls",
        ),
        per_task_rows,
    )
    write_csv(
        output / "d9_paired_outcomes.csv",
        ("paired_outcome", "count"),
        list(success["paired_outcome_2x2"].items()),
    )
    write_csv(
        output / "d9_primary_gates.csv",
        ("gate", "pass"),
        [(name, value) for name, value in result["gate_checks"].items()],
    )
    (output / "d9_result_overview.svg").write_text(
        _svg(result), encoding="utf-8"
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    checks = {
        "D9_formal_result_SHA_exact": True,
        "D9_status_and_all_18_primary_gates_PASS": True,
        "main_table_exported": "d9_main_results.csv" in artifacts,
        "per_task_table_exported": "d9_per_task_results.csv" in artifacts,
        "paired_outcome_table_exported": "d9_paired_outcomes.csv" in artifacts,
        "primary_gate_table_exported": "d9_primary_gates.csv" in artifacts,
        "vector_overview_exported": "d9_result_overview.svg" in artifacts,
        "no_new_model_fit_threshold_search_or_rollout": True,
    }
    if not all(checks.values()):
        raise PermissionError("D10 paper export checks failed")
    attestation_value = {
        "status": STATUS,
        "schema_version": "phase-route-vla.v3.d10-paper-analysis.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty_at_entry": False,
        "input": {"path": INPUT.as_posix(), "sha256": INPUT_SHA256},
        "output_directory": OUTPUT_DIR.as_posix(),
        "artifact_sha256": artifacts,
        "checks": checks,
        "access_ledger": {
            "frozen_D9_formal_results_opened": 1,
            "raw_D9C_rollouts_opened": 0,
            "raw_D9D_truth_payloads_opened": 0,
            "new_model_fits": 0,
            "new_threshold_searches": 0,
            "new_rollouts": 0,
            "CUDA_initialized": False,
        },
        "claim_boundary": {
            "paper_assets_add_new_experimental_evidence": False,
            "cross_suite_or_real_robot_generalization_authorized": False,
            "early_exit_failure_causality_authorized": False,
        },
    }
    attestation.write_text(
        json.dumps(attestation_value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    attestation.with_suffix(".sha256").write_text(
        f"{sha256_file(attestation)}  {attestation.name}\n", encoding="utf-8"
    )
    print(STATUS)


if __name__ == "__main__":
    main()
