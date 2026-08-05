"""Differentiable frozen-A1 action supervision for learnable visual aggregation.

The dynamic exit decision itself remains non-differentiable.  A cache record
provides the teacher's selected layer and the exact flow-matching initial
state; this module runs that layer once and propagates action loss only into
the learnable visual aggregator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from a1.config import ActivationCheckpointingStrategy


@dataclass
class FrozenA1DistillationOutput:
    normalized_action: torch.Tensor
    exit_layer: int
    fm_pos_offset: torch.Tensor


@dataclass
class FrozenA1ContextOutput:
    attn_key_values: Any
    exit_layer: int
    fm_pos_offset: torch.Tensor


def freeze_a1_for_action_distillation(model: nn.Module) -> nn.Module:
    """Freeze the teacher while leaving gradients with respect to its inputs."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def configure_frozen_a1_activation_checkpointing(
    model: nn.Module,
    strategy: str | None,
) -> str:
    """Configure recomputation for deep cached-exit distillation graphs.

    Frozen parameters still require transformer activations because gradients
    must flow from the action loss back to the learnable visual aggregator.
    Checkpointing those frozen blocks trades extra compute for substantially
    lower activation memory without changing the supervised action target.
    """

    normalized = "none" if strategy is None else str(strategy).strip().lower()
    if normalized == "none":
        return normalized
    try:
        checkpointing = ActivationCheckpointingStrategy(normalized)
    except ValueError as error:
        choices = ", ".join(item.value for item in ActivationCheckpointingStrategy)
        raise ValueError(
            f"unknown activation checkpointing strategy {strategy!r}; "
            f"expected one of: none, {choices}"
        ) from error
    setter = getattr(model, "set_activation_checkpointing", None)
    if not callable(setter):
        raise TypeError("frozen A1 model does not support activation checkpointing")
    setter(checkpointing)
    return checkpointing.value


def _optional_tensor(batch: Mapping[str, torch.Tensor], name: str):
    value = batch.get(name)
    if value is None or value.numel() == 0:
        return None
    return value


def select_cached_candidate_supervision(
    batch: dict[str, torch.Tensor],
    candidate_ordinal: int,
) -> dict[str, torch.Tensor]:
    """Select one cached candidate action while keeping a single FM graph."""

    roles = batch.get("fm_trace_roles")
    layers = batch.get("fm_trace_layers")
    inputs = batch.get("fm_trace_input_x")
    outputs = batch.get("fm_trace_output_action")
    if any(value is None for value in (roles, layers, inputs, outputs)):
        raise KeyError("candidate supervision requires v3 FM trace tensors")
    if roles.ndim != 2 or roles.shape[0] != 1:
        raise ValueError("candidate supervision currently requires batch size 1")
    if layers.shape != roles.shape or inputs.shape[:2] != roles.shape:
        raise ValueError("candidate trace tensors are not aligned")
    if outputs.shape != inputs.shape:
        raise ValueError("candidate trace inputs and outputs are not aligned")
    candidates = torch.nonzero(roles[0] == 1, as_tuple=False).flatten()
    if candidates.numel() < 1:
        raise ValueError("cache call has no candidate action trace")
    selected = int(candidates[candidate_ordinal % candidates.numel()].item())
    batch["teacher_exit_layer"] = layers[:, selected]
    batch["teacher_exit_input_x"] = inputs[:, selected]
    batch["teacher_action"] = outputs[:, selected]
    return batch


def frozen_a1_context_forward(
    model: nn.Module,
    aggregator: Optional[nn.Module],
    batch: Mapping[str, torch.Tensor],
    *,
    exit_layer: Optional[int] = None,
) -> FrozenA1ContextOutput:
    """Run the cached fixed-depth LLM prefix without solving the FM ODE."""

    required = (
        "input_ids",
        "projected_features",
        "image_input_idx",
        "instruction_summary",
        "action_proprio",
        "proprio_token_idx",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"distillation batch is missing: {', '.join(missing)}")
    if getattr(model.config, "action_head", None) != "flow_matching":
        raise ValueError("frozen A1 action distillation requires a flow_matching head")

    cached_layers = batch.get("teacher_exit_layer")
    if exit_layer is None:
        if cached_layers is None or cached_layers.numel() < 1:
            raise ValueError("teacher_exit_layer is required")
        unique_layers = torch.unique(cached_layers.detach().to(torch.int64))
        if unique_layers.numel() != 1:
            raise ValueError("one distillation batch cannot mix teacher exit layers")
        exit_layer = int(unique_layers.item())
    if not 0 <= exit_layer < int(model.config.n_layers):
        raise ValueError("teacher exit layer is outside the frozen A1 transformer")

    input_ids = batch["input_ids"]
    projected_features = batch["projected_features"]
    image_input_idx = batch["image_input_idx"]
    instruction_summary = batch["instruction_summary"]
    action_proprio = batch["action_proprio"]
    proprio_token_idx = batch["proprio_token_idx"]
    if projected_features.ndim != 4 or image_input_idx.shape != projected_features.shape[:3]:
        raise ValueError("cached projected features and image positions are not aligned")
    if input_ids.ndim != 2 or input_ids.shape[0] != projected_features.shape[0]:
        raise ValueError("input_ids must have shape [B, L]")

    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=_optional_tensor(batch, "attention_mask"),
        attention_bias=_optional_tensor(batch, "attention_bias"),
        response_mask=_optional_tensor(batch, "response_mask"),
        image_input_idx=image_input_idx,
        subsegment_ids=_optional_tensor(batch, "subsegment_ids"),
        position_ids=_optional_tensor(batch, "position_ids"),
        action_proprio=action_proprio,
        proprio_token_idx=proprio_token_idx,
        output_hidden_states=False,
        use_cache=True,
        exit_id=exit_layer,
        precomputed_projected_features=projected_features,
        learnable_vision_aggregator=aggregator,
        vision_instruction_summary=(
            instruction_summary if aggregator is not None else None
        ),
    )
    if outputs.exit_layer != exit_layer:
        raise RuntimeError(
            f"frozen A1 stopped at layer {outputs.exit_layer}, expected {exit_layer}"
        )
    if not outputs.attn_key_values or len(outputs.attn_key_values) != exit_layer + 1:
        raise RuntimeError("frozen A1 did not return the expected fixed-depth KV cache")
    if outputs.fm_pos_offset is None:
        raise RuntimeError("frozen A1 did not return its compacted FM position offset")

    return FrozenA1ContextOutput(
        attn_key_values=outputs.attn_key_values,
        exit_layer=exit_layer,
        fm_pos_offset=outputs.fm_pos_offset,
    )


def frozen_a1_action_forward(
    model: nn.Module,
    aggregator: Optional[nn.Module],
    batch: Mapping[str, torch.Tensor],
    *,
    exit_layer: Optional[int] = None,
) -> FrozenA1DistillationOutput:
    """Run one cached fixed-depth LLM prefix and one differentiable FM solve."""

    fm_input_x = batch.get("teacher_exit_input_x")
    if fm_input_x is None:
        raise KeyError("distillation batch is missing: teacher_exit_input_x")
    if fm_input_x.ndim != 3:
        raise ValueError("teacher_exit_input_x must have shape [B, H, A]")
    context = frozen_a1_context_forward(
        model,
        aggregator,
        batch,
        exit_layer=exit_layer,
    )

    action = model.predict_actions_flow_matching(
        context.attn_key_values,
        batch["action_proprio"],
        context.fm_pos_offset,
        input_x=fm_input_x,
    )
    return FrozenA1DistillationOutput(
        normalized_action=action,
        exit_layer=context.exit_layer,
        fm_pos_offset=context.fm_pos_offset,
    )


def frozen_a1_action_distillation_loss(
    output: FrozenA1DistillationOutput,
    teacher_action: torch.Tensor,
    *,
    first_step_weight: float = 1.0,
    gripper_weight: float = 1.0,
    gripper_transition_weight: float = 1.0,
    gripper_transition_threshold: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Risk-weightable Smooth-L1 plus interpretable action metrics."""

    predicted = output.normalized_action.float()
    target = teacher_action.float()
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and teacher actions must share shape [B, H, A]")
    if first_step_weight <= 0.0 or gripper_weight <= 0.0:
        raise ValueError("action loss weights must be positive")
    if gripper_transition_weight <= 0.0:
        raise ValueError("gripper transition weight must be positive")
    if gripper_transition_threshold < 0.0:
        raise ValueError("gripper transition threshold cannot be negative")

    element_loss = F.smooth_l1_loss(predicted, target, reduction="none")
    weights = torch.ones_like(element_loss)
    weights[:, 0] *= first_step_weight
    if predicted.shape[-1] >= 7:
        weights[..., 6:] *= gripper_weight
        if target.shape[1] > 1 and gripper_transition_weight != 1.0:
            transitions = gripper_transition_mask(
                target, gripper_transition_threshold
            )
            weights[transitions, :, 6:] *= gripper_transition_weight
    total = (element_loss * weights).sum() / weights.sum()
    parts = {
        "total": total.detach(),
        "mae": (predicted - target).abs().mean().detach(),
        "first_step_mae": (predicted[:, 0] - target[:, 0]).abs().mean().detach(),
    }
    if predicted.shape[-1] >= 7:
        parts.update(
            translation_mae=(predicted[..., :3] - target[..., :3]).abs().mean().detach(),
            rotation_mae=(predicted[..., 3:6] - target[..., 3:6]).abs().mean().detach(),
            gripper_mae=(predicted[..., 6:] - target[..., 6:]).abs().mean().detach(),
        )
    return total, parts


def gripper_transition_mask(
    action: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return one flag per action chunk containing a gripper transition."""

    if action.ndim != 3:
        raise ValueError("action must have shape [B, H, A]")
    if threshold < 0.0:
        raise ValueError("gripper transition threshold cannot be negative")
    if action.shape[1] < 2 or action.shape[2] < 7:
        return torch.zeros(
            action.shape[0], dtype=torch.bool, device=action.device
        )
    return (
        action[:, 1:, 6:] - action[:, :-1, 6:]
    ).abs().amax(dim=(1, 2)) >= threshold
