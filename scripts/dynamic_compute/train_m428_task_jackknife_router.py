"""Fit frozen M4.28 routers without producing sealed episode20--29 predictions."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_compute.train_m427_task_jackknife_router import main


if __name__ == "__main__":
    main("m428")
