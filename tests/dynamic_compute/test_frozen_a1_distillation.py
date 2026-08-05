from types import SimpleNamespace

import torch
from torch import nn

from a1.vla.dynamic_compute.frozen_a1_distillation import (
    configure_frozen_a1_activation_checkpointing,
    freeze_a1_for_action_distillation,
    frozen_a1_action_distillation_loss,
    frozen_a1_action_forward,
    frozen_a1_context_forward,
    gripper_transition_mask,
    select_cached_candidate_supervision,
)
from a1.vla.dynamic_compute.vision_aggregation import AggregatedVision


class _TinyAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, features, positions, instruction):
        del instruction
        selected = features[:, 0, :2] * self.scale
        selected_positions = positions[:, 0, :2]
        batch = features.shape[0]
        aggregated = AggregatedVision(
            features=selected,
            sequence_positions=selected_positions,
            valid_mask=torch.ones(batch, 2, dtype=torch.bool),
            crop_ids=torch.zeros(batch, 2, dtype=torch.long),
            source_counts=torch.full((batch,), 4, dtype=torch.long),
            original_counts=torch.full((batch,), 4, dtype=torch.long),
            bank_counts=torch.full((batch,), 2, dtype=torch.long),
            kept_counts=torch.full((batch,), 2, dtype=torch.long),
        )
        return SimpleNamespace(aggregated=aggregated)


class _TinyFrozenA1(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            action_head="flow_matching",
            n_layers=4,
        )
        self.activation_checkpointing_strategy = None

    def set_activation_checkpointing(self, strategy):
        self.activation_checkpointing_strategy = strategy

    def forward(self, **kwargs):
        aggregation = kwargs["learnable_vision_aggregator"](
            kwargs["precomputed_projected_features"],
            kwargs["image_input_idx"],
            kwargs["vision_instruction_summary"],
        ).aggregated
        exit_layer = kwargs["exit_id"]
        kv = (aggregation.features, aggregation.features)
        return SimpleNamespace(
            exit_layer=exit_layer,
            attn_key_values=[kv for _ in range(exit_layer + 1)],
            fm_pos_offset=(kwargs["input_ids"] != -1).sum(dim=1),
        )

    def predict_actions_flow_matching(
        self, attn_key_values, proprio, pos_offset, input_x=None
    ):
        del proprio, pos_offset
        return input_x + attn_key_values[-1][0].mean()


def _batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, -1]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "attention_bias": torch.empty(1, 0),
        "response_mask": torch.zeros(1, 4, dtype=torch.bool),
        "subsegment_ids": torch.empty(1, 0, dtype=torch.long),
        "position_ids": torch.arange(4).reshape(1, 4),
        "projected_features": torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
        "image_input_idx": torch.arange(4).reshape(1, 1, 4),
        "instruction_summary": torch.ones(1, 4),
        "action_proprio": torch.ones(1, 1, 8),
        "proprio_token_idx": torch.tensor([[2]]),
        "teacher_exit_input_x": torch.zeros(1, 8, 7),
        "teacher_exit_layer": torch.tensor([2]),
    }


def test_frozen_teacher_keeps_input_gradient_path_to_aggregator():
    teacher = freeze_a1_for_action_distillation(_TinyFrozenA1())
    aggregator = _TinyAggregator()
    batch = _batch()
    output = frozen_a1_action_forward(teacher, aggregator, batch)
    target = torch.zeros_like(output.normalized_action)
    loss, metrics = frozen_a1_action_distillation_loss(output, target)
    loss.backward()

    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert aggregator.scale.grad is not None
    assert torch.isfinite(aggregator.scale.grad)
    assert set(metrics) == {
        "total",
        "mae",
        "first_step_mae",
        "translation_mae",
        "rotation_mae",
        "gripper_mae",
    }


def test_context_forward_matches_action_forward_metadata():
    teacher = freeze_a1_for_action_distillation(_TinyFrozenA1())
    aggregator = _TinyAggregator()
    batch = _batch()

    context = frozen_a1_context_forward(teacher, aggregator, batch)
    action = frozen_a1_action_forward(teacher, aggregator, batch)

    assert context.exit_layer == action.exit_layer == 2
    assert len(context.attn_key_values) == 3
    torch.testing.assert_close(context.fm_pos_offset, action.fm_pos_offset)


def test_frozen_teacher_accepts_whole_layer_activation_checkpointing():
    teacher = freeze_a1_for_action_distillation(_TinyFrozenA1())

    configured = configure_frozen_a1_activation_checkpointing(
        teacher, "whole_layer"
    )

    assert configured == "whole_layer"
    assert teacher.activation_checkpointing_strategy.value == "whole_layer"


def test_distillation_rejects_mixed_exit_layers():
    teacher = freeze_a1_for_action_distillation(_TinyFrozenA1())
    aggregator = _TinyAggregator()
    batch = _batch()
    batch["input_ids"] = batch["input_ids"].repeat(2, 1)
    for name in (
        "attention_mask",
        "response_mask",
        "position_ids",
        "projected_features",
        "image_input_idx",
        "instruction_summary",
        "action_proprio",
        "proprio_token_idx",
        "teacher_exit_input_x",
    ):
        batch[name] = batch[name].repeat(2, *([1] * (batch[name].ndim - 1)))
    batch["teacher_exit_layer"] = torch.tensor([1, 2])

    try:
        frozen_a1_action_forward(teacher, aggregator, batch)
    except ValueError as error:
        assert "cannot mix" in str(error)
    else:
        raise AssertionError("expected mixed exit-layer validation to fail")


def test_candidate_trace_selection_uses_only_candidate_roles_and_wraps():
    batch = {
        "fm_trace_roles": torch.tensor([[0, 1, 1]], dtype=torch.uint8),
        "fm_trace_layers": torch.tensor([[10, 11, 13]]),
        "fm_trace_input_x": torch.stack(
            [torch.full((3, 8, 7), value) for value in (1.0,)]
        ),
        "fm_trace_output_action": torch.stack(
            [torch.arange(3 * 8 * 7, dtype=torch.float32).reshape(3, 8, 7)]
        ),
    }

    selected = select_cached_candidate_supervision(batch, 2)

    assert selected["teacher_exit_layer"].tolist() == [11]
    torch.testing.assert_close(
        selected["teacher_action"], batch["fm_trace_output_action"][:, 1]
    )


def test_transition_weighted_loss_penalizes_early_gripper_timing_error():
    target = torch.zeros(1, 8, 7)
    target[:, -1, 6] = 1.0
    predicted = target.clone()
    predicted[:, -3:-1, 6] = 1.0
    output = SimpleNamespace(normalized_action=predicted)

    plain, _ = frozen_a1_action_distillation_loss(output, target)
    weighted, _ = frozen_a1_action_distillation_loss(
        output,
        target,
        gripper_weight=2.0,
        gripper_transition_weight=8.0,
    )

    assert weighted > plain


def test_gripper_transition_mask_handles_transition_and_non_gripper_actions():
    actions = torch.zeros(2, 4, 7)
    actions[1, -1, 6] = 1.0

    assert gripper_transition_mask(actions, 0.5).tolist() == [False, True]
    assert not gripper_transition_mask(actions[..., :6], 0.5).any()

    try:
        gripper_transition_mask(actions, -0.1)
    except ValueError as error:
        assert "cannot be negative" in str(error)
    else:
        raise AssertionError("negative transition threshold was accepted")
