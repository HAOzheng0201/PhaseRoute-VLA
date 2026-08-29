"""Load Stage 10 CPU contracts without executing the top-level ``a1`` package."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load Stage 10 contract: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


CONTRACT = _load(
    "_phase_route_stage10_contract",
    REPO_ROOT / "a1/vla/dynamic_compute/route_first_stage10.py",
)
ACTIVE = _load(
    "_phase_route_stage10_active_contract",
    REPO_ROOT / "a1/vla/dynamic_compute/route_first_stage10_active.py",
)


__all__ = ["ACTIVE", "CONTRACT", "REPO_ROOT"]
