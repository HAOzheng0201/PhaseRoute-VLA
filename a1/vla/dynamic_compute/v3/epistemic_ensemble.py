"""Filesystem-light protocol primitives for frozen V3-D7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


D7_SCHEMA_VERSION = "phase-route-vla.v3.d7-epistemic-ensemble-contract.v1"
D7_STATUS = "D7_EPISTEMIC_ENSEMBLE_CONTRACT_FROZEN"
D7_CONTRACT_RELATIVE_PATH = Path(
    "configs/research/v3/joint_reliability/d7_epistemic_ensemble_contract.json"
)
D7_CONTRACT_SHA256 = (
    "7e1f8934e33ae33493b950eabc1142c1f6cd7103ef7b4ad735d6c8b13a5afdea"
)
D7_HEAD_COUNT = 5
D7_DELETE_GROUP_COUNT = 4
D7_FEATURE_PARAMETERS = 970
D7_FITS_PER_OUTER = 260
D7_TOTAL_FITS = 4680
D7_MIN_EARLY_FRACTION = 0.10
D7_MAX_FALSE_FULL_CLUSTERS = 3
D7_MIN_NONDEGENERATE_FRACTION = 0.01
D7_MIN_HEAD_RANGE = 1.0e-6


class D7ProtocolError(ValueError):
    """Raised whenever D7 protocol geometry or metadata differs."""


def delete_group_index(episode_index: torch.Tensor) -> torch.Tensor:
    if (
        episode_index.device.type != "cpu"
        or episode_index.dtype != torch.long
        or episode_index.ndim != 1
        or episode_index.numel() == 0
        or not bool(((episode_index >= 12) & (episode_index <= 29)).all())
    ):
        raise D7ProtocolError("D7 episode index must be CPU int64 [N] in 12--29")
    return torch.remainder(episode_index - 12, D7_DELETE_GROUP_COUNT).contiguous()


def head_fit_masks(
    base_fit_mask: torch.Tensor, episode_index: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    rows = int(episode_index.numel())
    if (
        base_fit_mask.device.type != "cpu"
        or base_fit_mask.dtype != torch.bool
        or base_fit_mask.shape != (rows,)
        or int(base_fit_mask.sum()) < 8
    ):
        raise D7ProtocolError("D7 base fit mask must be CPU bool [N]")
    group = delete_group_index(episode_index)
    masks = [base_fit_mask.contiguous()]
    for value in range(D7_DELETE_GROUP_COUNT):
        mask = (base_fit_mask & (group != value)).contiguous()
        if int(mask.sum()) < 4:
            raise D7ProtocolError("D7 delete-group head fit partition is too small")
        masks.append(mask)
    result = tuple(masks)
    if len(result) != D7_HEAD_COUNT:
        raise D7ProtocolError("D7 head count differs")
    included = torch.stack(result).long().sum(dim=0)
    if not bool((included[~base_fit_mask] == 0).all()) or not bool(
        (included[base_fit_mask] == 4).all()
    ):
        raise D7ProtocolError("D7 delete-group coverage differs")
    return result


def ensemble_scores(prediction: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if (
        prediction.device.type != "cpu"
        or prediction.ndim != 3
        or prediction.shape[0] != D7_HEAD_COUNT
        or prediction.shape[2] != 2
        or not prediction.is_floating_point()
        or not bool(torch.isfinite(prediction).all())
        or not bool(((prediction > 0.0) & (prediction < 1.0)).all())
    ):
        raise D7ProtocolError("D7 prediction must be finite CPU [5,N,2]")
    full = prediction[:, :, 0].double()
    upper = full.max(dim=0).values.contiguous()
    gripper = prediction[0, :, 1].double().contiguous()
    head_range = (full.max(dim=0).values - full.min(dim=0).values).contiguous()
    return upper, gripper, head_range


def load_d7_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    path = root / D7_CONTRACT_RELATIVE_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != D7_CONTRACT_SHA256:
        raise D7ProtocolError("D7 frozen contract SHA-256 differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D7ProtocolError("D7 contract cannot be read") from error
    if not isinstance(value, Mapping):
        raise D7ProtocolError("D7 contract must be an object")
    contract = dict(value)
    if (
        contract.get("schema_version") != D7_SCHEMA_VERSION
        or contract.get("status") != D7_STATUS
        or contract.get("scope", {}).get("fresh_confirmation_claim_allowed") is not False
        or contract.get("epistemic_ensemble", {}).get("head_count") != D7_HEAD_COUNT
        or contract.get("epistemic_ensemble", {}).get(
            "trainable_feature_parameter_count"
        )
        != D7_FEATURE_PARAMETERS
        or contract.get("nested_oof", {}).get("fits_per_outer")
        != D7_FITS_PER_OUTER
        or contract.get("nested_oof", {}).get("total_model_fits") != D7_TOTAL_FITS
        or contract.get("development_selection_criteria", {}).get(
            "minimum_early_exit_call_fraction"
        )
        != D7_MIN_EARLY_FRACTION
        or contract.get("authorization", {}).get("independent_test_authorized")
        is not False
    ):
        raise D7ProtocolError("D7 frozen contract semantics differ")
    return contract


__all__ = [
    "D7_CONTRACT_RELATIVE_PATH",
    "D7_CONTRACT_SHA256",
    "D7_FEATURE_PARAMETERS",
    "D7_FITS_PER_OUTER",
    "D7_HEAD_COUNT",
    "D7_MAX_FALSE_FULL_CLUSTERS",
    "D7_MIN_EARLY_FRACTION",
    "D7_MIN_HEAD_RANGE",
    "D7_MIN_NONDEGENERATE_FRACTION",
    "D7_TOTAL_FITS",
    "D7ProtocolError",
    "delete_group_index",
    "ensemble_scores",
    "head_fit_masks",
    "load_d7_contract",
]
