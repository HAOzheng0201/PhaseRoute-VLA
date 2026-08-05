import pytest
import torch
from torch import nn
from types import SimpleNamespace

from a1.model import BufferCache
from a1.vla.affordvla_early_exit import AffordVLAEarlyExit
from a1.vla.dynamic_compute.vision_aggregation import (
    StaticVisionAggregationConfig,
    aggregate_projected_vision,
    compact_multimodal_sequence,
    rank_token_bank,
)
from a1.vla.dynamic_compute.learnable_vision_aggregation import (
    LearnableVisionAggregationConfig,
    LearnableVisionAggregator,
)


def _vision_fixture(batch_size=1, crop_count=4, tokens_per_crop=144, hidden_dim=8):
    features = torch.arange(
        batch_size * crop_count * tokens_per_crop * hidden_dim,
        dtype=torch.float32,
    ).reshape(batch_size, crop_count, tokens_per_crop, hidden_dim)
    positions = torch.arange(
        crop_count * tokens_per_crop,
        dtype=torch.long,
    ).reshape(1, crop_count, tokens_per_crop)
    positions = positions.expand(batch_size, -1, -1).clone() + 5
    return features, positions


def test_config_defaults_to_disabled_and_rejects_invalid_budgets():
    assert StaticVisionAggregationConfig().enabled is False
    with pytest.raises(ValueError, match="bank_tokens"):
        StaticVisionAggregationConfig(enabled=True, keep_tokens=64, bank_tokens=32)
    with pytest.raises(ValueError, match="min_tokens_per_crop"):
        StaticVisionAggregationConfig(enabled=True, min_tokens_per_crop=0)


def test_keep_all_preserves_projected_tokens_bitwise_and_in_sequence_order():
    features, positions = _vision_fixture(crop_count=2, tokens_per_crop=16)
    result = aggregate_projected_vision(
        features,
        positions,
        StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=32,
            min_tokens_per_crop=2,
        ),
    )

    assert result.compression_applied is False
    assert torch.equal(result.features, features.reshape(1, 32, 8))
    assert torch.equal(result.sequence_positions, positions.reshape(1, 32))
    assert result.original_counts.tolist() == [32]
    assert result.source_counts.tolist() == [32]
    assert result.bank_counts.tolist() == [32]
    assert result.kept_counts.tolist() == [32]


def test_static_64_token_pool_has_exact_shape_and_protects_every_crop():
    features, positions = _vision_fixture()
    result = aggregate_projected_vision(
        features,
        positions,
        StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=64,
            min_tokens_per_crop=4,
        ),
    )

    assert result.features.shape == (1, 64, 8)
    assert result.sequence_positions.shape == (1, 64)
    assert result.valid_mask.all()
    assert result.compression_applied is True
    assert result.original_counts.tolist() == [576]
    assert result.source_counts.tolist() == [576]
    assert result.bank_counts.tolist() == [64]
    assert result.kept_counts.tolist() == [64]
    for crop_id in range(4):
        assert int((result.crop_ids[0] == crop_id).sum()) == 16
    assert torch.all(result.sequence_positions[:, 1:] > result.sequence_positions[:, :-1])


def test_vectorized_spatial_pool_matches_exact_block_means():
    features = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1)
    positions = torch.arange(16, dtype=torch.long).reshape(1, 1, 16) + 5

    result = aggregate_projected_vision(
        features,
        positions,
        StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=4,
            min_tokens_per_crop=1,
        ),
    )

    torch.testing.assert_close(
        result.features[0, :, 0],
        torch.tensor([2.5, 4.5, 10.5, 12.5]),
    )
    assert result.sequence_positions.tolist() == [[5, 7, 13, 15]]


def test_fixed_bank_produces_nested_visual_position_sets():
    features, positions = _vision_fixture()
    result_32 = aggregate_projected_vision(
        features,
        positions,
        StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=32,
            bank_tokens=96,
            min_tokens_per_crop=4,
        ),
    )
    result_64 = aggregate_projected_vision(
        features,
        positions,
        StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=64,
            bank_tokens=96,
            min_tokens_per_crop=4,
        ),
    )

    positions_32 = set(result_32.sequence_positions[0].tolist())
    positions_64 = set(result_64.sequence_positions[0].tolist())
    assert positions_32 < positions_64
    for crop_id in range(4):
        assert int((result_32.crop_ids[0] == crop_id).sum()) >= 4
        assert int((result_64.crop_ids[0] == crop_id).sum()) >= 4


def test_rank_token_bank_returns_nested_prefix_order():
    features = torch.tensor(
        [[[1.0], [4.0], [2.0], [8.0], [3.0], [7.0]]]
    )
    valid_mask = torch.ones((1, 6), dtype=torch.bool)
    crop_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

    ranking = rank_token_bank(features, valid_mask, crop_ids, min_tokens_per_crop=1)

    first_four = set(ranking[0, :4].tolist())
    first_six = set(ranking[0, :6].tolist())
    assert first_four < first_six
    assert {int(crop_ids[0, index]) for index in first_four} == {0, 1}


def test_nonfinite_projected_feature_is_rejected_before_compaction():
    features, positions = _vision_fixture(crop_count=1, tokens_per_crop=4)
    features[0, 0, 2, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        aggregate_projected_vision(
            features,
            positions,
            StaticVisionAggregationConfig(
                enabled=True,
                keep_tokens=2,
                min_tokens_per_crop=1,
            ),
        )


def test_overlapping_crop_positions_are_canonicalized_with_last_write_wins():
    features = torch.tensor(
        [[[[1.0], [2.0], [3.0], [4.0]], [[10.0], [20.0], [30.0], [40.0]]]]
    )
    positions = torch.tensor([[[2, 3, 4, 5], [2, 3, 4, 5]]])

    result = aggregate_projected_vision(
        features,
        positions,
        StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=4,
            min_tokens_per_crop=1,
        ),
    )

    assert result.source_counts.tolist() == [8]
    assert result.original_counts.tolist() == [4]
    assert result.bank_counts.tolist() == [4]
    assert result.kept_counts.tolist() == [4]
    assert torch.equal(result.sequence_positions, torch.tensor([[2, 3, 4, 5]]))
    assert torch.equal(result.features, torch.tensor([[[10.0], [20.0], [30.0], [40.0]]]))


def test_compaction_gathers_every_sequence_field_and_remaps_positions():
    embeddings = torch.arange(10 * 3, dtype=torch.float32).reshape(1, 10, 3)
    input_ids = torch.arange(10).reshape(1, 10)
    original_visual = torch.tensor([[[2, 3], [6, 7]]])
    kept_visual = torch.tensor([[3, 6]])
    attention_mask = torch.ones((1, 10), dtype=torch.bool)
    attention_bias = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10)
    response_mask = torch.arange(10).reshape(1, 10) % 2
    subsegment_ids = torch.arange(10).reshape(1, 10) // 3
    position_ids = torch.arange(10).reshape(1, 10) + 100

    compacted = compact_multimodal_sequence(
        embeddings,
        input_ids,
        original_visual,
        kept_visual,
        attention_mask=attention_mask,
        attention_bias=attention_bias,
        response_mask=response_mask,
        subsegment_ids=subsegment_ids,
        position_ids=position_ids,
        proprio_token_idx=torch.tensor([[8]]),
        append_last_valid_logits=torch.tensor([9]),
    )

    expected_old_positions = torch.tensor([0, 1, 3, 4, 5, 6, 8, 9])
    assert compacted.embeddings.shape == (1, 8, 3)
    assert torch.equal(compacted.input_ids[0], input_ids[0, expected_old_positions])
    assert torch.equal(
        compacted.embeddings[0], embeddings[0, expected_old_positions]
    )
    assert torch.equal(
        compacted.response_mask[0], response_mask[0, expected_old_positions]
    )
    assert torch.equal(
        compacted.subsegment_ids[0], subsegment_ids[0, expected_old_positions]
    )
    assert torch.equal(
        compacted.position_ids[0], position_ids[0, expected_old_positions]
    )
    expected_bias = attention_bias[0, 0][expected_old_positions][
        :, expected_old_positions
    ]
    assert torch.equal(compacted.attention_bias[0, 0], expected_bias)
    assert compacted.image_input_idx.tolist() == [[2, 5]]
    assert compacted.proprio_token_idx.tolist() == [[6]]
    assert compacted.append_last_valid_logits.tolist() == [7]
    assert compacted.lengths.tolist() == [8]


def test_keep_all_compaction_is_bitwise_identity():
    embeddings = torch.randn(2, 12, 5)
    input_ids = torch.arange(12).repeat(2, 1)
    visual = torch.tensor([[[2, 4, 7]], [[2, 4, 7]]])
    attention_mask = torch.ones((2, 12), dtype=torch.bool)
    position_ids = torch.arange(12).repeat(2, 1)

    compacted = compact_multimodal_sequence(
        embeddings,
        input_ids,
        visual,
        visual,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    assert torch.equal(compacted.embeddings, embeddings)
    assert torch.equal(compacted.input_ids, input_ids)
    assert torch.equal(compacted.attention_mask, attention_mask)
    assert torch.equal(compacted.position_ids, position_ids)
    assert compacted.lengths.tolist() == [12, 12]


def test_compaction_handles_variable_visual_counts_with_tail_padding():
    embeddings = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8, 1)
    input_ids = torch.arange(8).repeat(2, 1)
    original_visual = torch.tensor([[[1, 2, 3, 4]], [[1, 2, -1, -1]]])
    kept_visual = torch.tensor([[1, 3], [2, -1]])

    compacted = compact_multimodal_sequence(
        embeddings,
        input_ids,
        original_visual,
        kept_visual,
    )

    assert compacted.lengths.tolist() == [6, 7]
    assert compacted.input_ids.shape == (2, 7)
    assert compacted.input_ids[0, -1].item() == -1
    assert compacted.image_input_idx.tolist() == [[1, 2], [1, -1]]


def test_compaction_accepts_overlapping_original_crop_positions():
    embeddings = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    input_ids = torch.arange(8).reshape(1, 8)

    compacted = compact_multimodal_sequence(
        embeddings,
        input_ids,
        torch.tensor([[[1, 2, 3], [1, 2, 3]]], dtype=torch.int32),
        torch.tensor([[1, 3]], dtype=torch.int32),
        proprio_token_idx=torch.tensor([[6]], dtype=torch.int32),
    )

    assert compacted.input_ids.tolist() == [[0, 1, 3, 4, 5, 6, 7]]
    assert compacted.image_input_idx.tolist() == [[1, 2]]
    assert compacted.proprio_token_idx.tolist() == [[5]]


def test_compaction_can_reindex_position_ids_when_requested():
    embeddings = torch.randn(1, 6, 2)
    input_ids = torch.tensor([[10, 11, 12, 13, 14, -1]])
    position_ids = torch.tensor([[20, 21, 22, 23, 24, 0]])

    compacted = compact_multimodal_sequence(
        embeddings,
        input_ids,
        torch.tensor([[[1, 2]]]),
        torch.tensor([[2]]),
        position_ids=position_ids,
        preserve_position_ids=False,
    )

    assert compacted.input_ids.tolist() == [[10, 12, 13, 14, -1]]
    assert compacted.position_ids.tolist() == [[0, 1, 2, 3, 3]]


class _IdentityBlock(nn.Module):
    def forward(
        self,
        x,
        attention_bias=None,
        position_ids=None,
        drop_mask=None,
        layer_past=None,
        use_cache=False,
    ):
        del attention_bias, position_ids, drop_mask, layer_past, use_cache
        return x, None


class _FixedVisionBackbone(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.register_buffer("features", features)

    def forward(self, images, image_masks=None):
        del images, image_masks
        return self.features.clone()


def _tiny_early_exit_model():
    model = AffordVLAEarlyExit.__new__(AffordVLAEarlyExit)
    nn.Module.__init__(model)
    hidden_dim = 4
    model.config = SimpleNamespace(
        n_layers=1,
        use_position_ids=True,
        use_proprio=False,
        rope=True,
        d_model=hidden_dim,
        normalize_input_embeds=False,
        llm_causal_attention=True,
        block_group_size=1,
        action_head="l1_regression",
        num_actions_chunk=1,
        action_dim=1,
    )
    model.action_head_type = "l1_regression"
    model.proprio_projector = None
    model.transformer = nn.Module()
    model.transformer.wte = nn.Embedding(32, hidden_dim)
    model.transformer.emb_drop = nn.Identity()
    model.transformer.blocks = nn.ModuleList([_IdentityBlock()])
    model.transformer.ln_f = nn.Identity()
    features = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, hidden_dim)
    model.vision_backbone = _FixedVisionBackbone(features)
    model.activation_checkpointing_strategy = None
    model._AffordVLAEarlyExit__cache = BufferCache()
    model.eval()
    return model


def _tiny_forward(model, config=None):
    return model.forward(
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        images=torch.zeros(1),
        image_input_idx=torch.tensor([[[2, 3, 4, 5]]]),
        position_ids=torch.arange(8).reshape(1, 8),
        proprio_token_idx=torch.tensor([[6]]),
        vision_aggregation_config=config,
    )


def test_model_disabled_and_keep_all_paths_are_bitwise_equivalent():
    model = _tiny_early_exit_model()

    baseline = _tiny_forward(model)
    disabled = _tiny_forward(model, StaticVisionAggregationConfig(enabled=False))
    keep_all = _tiny_forward(
        model,
        StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=4,
            min_tokens_per_crop=1,
        ),
    )

    assert torch.equal(baseline.last_hidden_state, disabled.last_hidden_state)
    assert torch.equal(baseline.last_hidden_state, keep_all.last_hidden_state)
    assert baseline.last_hidden_state.shape == (1, 8, 4)


def test_model_precomputed_projected_features_match_live_backbone_path():
    model = _tiny_early_exit_model()
    baseline = _tiny_forward(model)
    precomputed = model.forward(
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        precomputed_projected_features=model.vision_backbone.features.clone(),
        image_input_idx=torch.tensor([[[2, 3, 4, 5]]]),
        position_ids=torch.arange(8).reshape(1, 8),
        proprio_token_idx=torch.tensor([[6]]),
    )

    assert torch.equal(baseline.last_hidden_state, precomputed.last_hidden_state)


def test_model_learnable_aggregation_backpropagates_from_llm_output():
    model = _tiny_early_exit_model()
    aggregator = LearnableVisionAggregator(
        LearnableVisionAggregationConfig(
            hidden_dim=4,
            attention_dim=4,
            output_tokens=2,
            num_heads=1,
            max_crops=1,
            max_patches_per_crop=4,
            min_tokens_per_crop=1,
        )
    )
    output = model.forward(
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        precomputed_projected_features=model.vision_backbone.features.clone(),
        image_input_idx=torch.tensor([[[2, 3, 4, 5]]]),
        position_ids=torch.arange(8).reshape(1, 8),
        proprio_token_idx=torch.tensor([[6]]),
        learnable_vision_aggregator=aggregator,
        vision_instruction_summary=torch.ones(1, 4),
    )

    assert output.last_hidden_state.shape == (1, 6, 4)
    output.last_hidden_state.square().mean().backward()
    assert aggregator.queries.grad is not None
    assert torch.isfinite(aggregator.queries.grad).all()


def test_phase_callback_installs_plan_before_learnable_vision_routing():
    model = _tiny_early_exit_model()
    call_order = []

    class _OrderCheckingAggregator(nn.Module):
        def forward(self, features, positions, instruction_summary):
            assert call_order == ["phase"]
            assert instruction_summary.shape == (1, 4)
            call_order.append("aggregator")
            return aggregate_projected_vision(
                features,
                positions,
                StaticVisionAggregationConfig(
                    enabled=True,
                    keep_tokens=2,
                    min_tokens_per_crop=1,
                    fail_open=False,
                ),
            )

    output = model.forward(
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        precomputed_projected_features=model.vision_backbone.features.clone(),
        image_input_idx=torch.tensor([[[2, 3, 4, 5]]]),
        position_ids=torch.arange(8).reshape(1, 8),
        proprio_token_idx=torch.tensor([[6]]),
        phase_signal_callback=lambda payload: call_order.append("phase"),
        learnable_vision_aggregator=_OrderCheckingAggregator(),
        vision_instruction_summary=torch.ones(1, 4),
    )

    assert call_order == ["phase", "aggregator"]
    assert output.last_hidden_state.shape == (1, 6, 4)


def test_model_compressed_path_reduces_real_llm_sequence_length():
    model = _tiny_early_exit_model()

    compressed = _tiny_forward(
        model,
        StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=2,
            min_tokens_per_crop=1,
        ),
    )

    assert compressed.last_hidden_state.shape == (1, 6, 4)


def test_model_aggregation_error_is_fail_open_by_default():
    model = _tiny_early_exit_model()
    baseline = _tiny_forward(model)

    fail_open = _tiny_forward(
        model,
        StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=1,
            min_tokens_per_crop=2,
        ),
    )
    assert torch.equal(baseline.last_hidden_state, fail_open.last_hidden_state)

    with pytest.raises(ValueError, match="per-crop minimum"):
        _tiny_forward(
            model,
            StaticVisionAggregationConfig(
                enabled=True,
                keep_tokens=1,
                min_tokens_per_crop=2,
                fail_open=False,
            ),
        )
