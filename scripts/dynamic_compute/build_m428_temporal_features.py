"""Frozen M4.28 entry point for the 30-episode temporal feature protocol."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_compute.build_m426_temporal_features import main


if __name__ == "__main__":
    if "--protocol" in sys.argv or "--expected-seed" in sys.argv:
        raise SystemExit("M4.28 wrapper freezes protocol=m428 and seed=20261228")
    sys.argv[1:1] = ["--protocol", "m428", "--expected-seed", "20261228"]
    main()
