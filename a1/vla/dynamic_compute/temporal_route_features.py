"""Causal temporal feature helpers for M4.25b route prediction."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np


M425B_FEATURE_SCHEMA_VERSION = "phase-route-vla.m425b-temporal-route-features.v1"
_EPISODE_SUFFIX = re.compile(r":episode([0-9]+)$")
M425B_ROUTE_LAYERS = (11, 13, 27)
RAW_A1_EXIT_LAYERS = tuple(range(1, 28, 2))


def canonical_teacher_route(raw_exit_layer: int) -> int:
    """Map an A1 exit to the first M4.25b route that is not shallower.

    A1 can exit at every odd transformer layer, while the causal router can
    only make decisions at layers 11 and 13 and otherwise runs to layer 27.
    Taking the ceiling on that execution lattice preserves the fail-closed
    ordering: the derived target is never shallower than the teacher exit.
    """

    layer = int(raw_exit_layer)
    if layer not in RAW_A1_EXIT_LAYERS:
        raise ValueError(f"unsupported raw A1 exit layer: {layer}")
    if layer <= 11:
        return 11
    if layer <= 13:
        return 13
    return 27


def parse_episode_index(episode_id: str) -> int:
    match = _EPISODE_SUFFIX.search(str(episode_id))
    if match is None:
        raise ValueError(f"episode ID has no canonical suffix: {episode_id}")
    return int(match.group(1))


def right_aligned_history(
    history: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    history_len: int,
    proprio_dim: int,
    action_horizon: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return padded past-only proprio/action windows and their validity mask."""

    if min(history_len, proprio_dim, action_horizon, action_dim) < 1:
        raise ValueError("history dimensions must be positive")
    if len(history) > history_len:
        history = history[-history_len:]
    proprio = np.zeros((history_len, proprio_dim), dtype=np.float32)
    action = np.zeros(
        (history_len, action_horizon, action_dim), dtype=np.float32
    )
    mask = np.zeros((history_len,), dtype=np.bool_)
    start = history_len - len(history)
    for offset, (past_proprio, past_action) in enumerate(history, start=start):
        past_proprio = np.asarray(past_proprio, dtype=np.float32)
        past_action = np.asarray(past_action, dtype=np.float32)
        if past_proprio.shape != (proprio_dim,):
            raise ValueError("history proprio has an invalid shape")
        if past_action.shape != (action_horizon, action_dim):
            raise ValueError("history action has an invalid shape")
        if not np.isfinite(past_proprio).all() or not np.isfinite(past_action).all():
            raise ValueError("history contains a non-finite value")
        proprio[offset] = past_proprio
        action[offset] = past_action
        mask[offset] = True
    return proprio, action, mask
