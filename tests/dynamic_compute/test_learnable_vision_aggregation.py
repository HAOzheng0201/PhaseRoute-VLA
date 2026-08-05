import torch

from a1.vla.dynamic_compute.learnable_vision_aggregation import (
    LearnableEFAWarmupModel,
    LearnableVisionAggregationConfig,
    LearnableVisionAggregator,
    efa_warmup_loss,
    reparameterize_residual_scale,
)


def _inputs(batch_size=2, hidden_dim=16):
    torch.manual_seed(7)
    features = torch.randn(batch_size, 5, 16, hidden_dim)
    # Four source crops map in overlapping pairs to 32 unique LLM slots.
    positions = torch.full((batch_size, 5, 16), -1, dtype=torch.int32)
    positions[:, 0] = torch.arange(16)
    positions[:, 1] = torch.arange(16, 32)
    positions[:, 2] = torch.arange(16)
    positions[:, 3] = torch.arange(16, 32)
    instruction = torch.randn(batch_size, hidden_dim)
    proprio = torch.randn(batch_size, 8)
    action = torch.randn(batch_size, 8, 7)
    return features, positions, instruction, proprio, action


def _config():
    return LearnableVisionAggregationConfig(
        hidden_dim=16,
        attention_dim=8,
        output_tokens=16,
        num_heads=2,
        max_crops=5,
        max_patches_per_crop=16,
        min_tokens_per_crop=2,
        residual_gate_init=0.01,
    )


def test_learnable_aggregator_uses_all_valid_source_crops_and_masks_padding():
    features, positions, instruction, _, _ = _inputs()
    model = LearnableVisionAggregator(_config())

    output = model(features, positions, instruction)

    assert output.aggregated.features.shape == (2, 16, 16)
    assert output.attention_weights.shape == (2, 16, 80)
    attention = output.attention_weights.reshape(2, 16, 5, 16)
    assert torch.all(attention[:, :, :4].sum(dim=(1, 3)) > 0)
    torch.testing.assert_close(attention[:, :, 4], torch.zeros_like(attention[:, :, 4]))
    assert output.aggregated.source_counts.tolist() == [64, 64]
    assert output.aggregated.original_counts.tolist() == [32, 32]


def test_nested_prefix_positions_are_subsets():
    features, positions, instruction, _, _ = _inputs(batch_size=1)
    model = LearnableVisionAggregator(_config()).eval()

    positions8 = set(model(features, positions, instruction, keep_tokens=8).aggregated.sequence_positions[0].tolist())
    positions12 = set(model(features, positions, instruction, keep_tokens=12).aggregated.sequence_positions[0].tolist())
    positions16 = set(model(features, positions, instruction, keep_tokens=16).aggregated.sequence_positions[0].tolist())

    assert positions8 < positions12 < positions16


def test_warmup_loss_is_finite_and_backpropagates_into_aggregator():
    features, positions, instruction, proprio, action = _inputs()
    model = LearnableEFAWarmupModel(_config())
    output = model(features, positions, instruction, proprio)
    loss, parts = efa_warmup_loss(output, features, positions, action)

    assert torch.isfinite(loss)
    assert set(parts) == {"action", "crop", "anchor", "total"}
    loss.backward()
    assert model.aggregator.queries.grad is not None
    assert torch.isfinite(model.aggregator.queries.grad).all()
    assert model.aggregator.output_projection.weight.grad is not None


def test_invalid_keep_budget_is_rejected():
    features, positions, instruction, _, _ = _inputs(batch_size=1)
    model = LearnableVisionAggregator(_config())

    try:
        model(features, positions, instruction, keep_tokens=3)
    except ValueError as error:
        assert "minimum" in str(error)
    else:
        raise AssertionError("expected crop minimum validation to fail")


def test_residual_scale_reparameterization_preserves_output_and_raises_gate():
    features, positions, instruction, _, _ = _inputs(batch_size=1)
    model = LearnableVisionAggregator(_config()).eval()
    before = model(features, positions, instruction).aggregated.features.detach()

    metadata = reparameterize_residual_scale(model, 0.2)
    after = model(features, positions, instruction).aggregated.features.detach()

    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
    assert abs(metadata["original_scale"] - 0.01) < 1e-6
    assert abs(metadata["actual_scale"] - 0.2) < 1e-6
    assert metadata["gradient_scale_gain"] > 19.0
