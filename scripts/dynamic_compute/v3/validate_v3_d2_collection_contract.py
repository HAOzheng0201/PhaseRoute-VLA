#!/usr/bin/env python3
"""Validate the label-independent V3-D2 collection contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    collection_contract,
    collection_contract_sha256,
    load_development_selection,
    stream_sha256,
    validate_frozen_d2_inputs,
)


CONTRACT_PATH = REPO_ROOT / "configs/research/v3/gripper_v2/d2_collection_contract.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observed = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    expected = collection_contract()
    if observed != expected:
        raise SystemExit("V3-D2 collection contract differs from implementation")
    selection = load_development_selection(REPO_ROOT)
    frozen = validate_frozen_d2_inputs(REPO_ROOT)
    result = {
        "status": "PASS_V3_D2_COLLECTION_CONTRACT",
        "contract_path": str(CONTRACT_PATH.relative_to(REPO_ROOT)),
        "contract_file_sha256": stream_sha256(CONTRACT_PATH),
        "contract_canonical_sha256": collection_contract_sha256(),
        "development_keys": len(selection),
        "first_key": selection[0].group_key,
        "last_key": selection[-1].group_key,
        "calibration_or_test_payload_opened": False,
        "legacy_c361_row_payload_opened": False,
        "frozen_inputs": frozen,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        expected = (
            REPO_ROOT / "results/v3/v3_d2_collection_contract_validation.json"
        ).resolve()
        if output != expected:
            raise SystemExit("V3-D2 validation output path differs")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() or output.with_suffix(".sha256").exists():
            raise FileExistsError("V3-D2 refuses to overwrite validation evidence")
        output.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        output.with_suffix(".sha256").write_text(
            f"{digest}  {output.name}\n", encoding="utf-8"
        )
    print(text, end="")


if __name__ == "__main__":
    main()
