import pytest
import torch
from hypothesis import assume, given, strategies as st
from strategies import attention_shapes, batch_time_shapes, hypothesis_settings

from models.shared.block import Block, block_flops
from models.shared.rope import RotaryEmbedding


def make_block(d_model, n_heads, **overrides):
    rope = overrides.pop("rope", None)
    if rope is None and n_heads > 0 and d_model % n_heads == 0:
        rope = RotaryEmbedding(d_model // n_heads, 4096, 10_000.0)
    config = {
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "multiple_of": 32,
        "rope": rope,
        "norm_eps": 1e-6,
    }
    config.update(overrides)
    return Block(d_model, n_heads, **config)


def run_block(block, x, *, mask=None, is_causal=True, cache=None):
    return block(x, mask=mask, is_causal=is_causal, cache=cache)


class TestFlopAccounting:
    @given(
        attention_shapes(),
        st.floats(min_value=0.5, max_value=8.0),
        st.sampled_from([16, 32, 64]),
    )
    @hypothesis_settings
    def test_flop_equation_accuracy(self, attn_shape, mlp_ratio, multiple_of):
        d_model, n_heads = attn_shape

        block = make_block(
            d_model=d_model,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            multiple_of=multiple_of,
        )
        calculated_flops = block_flops(d_model, mlp_ratio, multiple_of)

        total_matrix_params = sum(p.numel() for p in block.parameters() if p.dim() > 1)
        actual_flops_per_token = total_matrix_params * 2

        assert calculated_flops == actual_flops_per_token, (
            f"FLOP calculation mismatch. Expected {calculated_flops}, "
            f"computed {actual_flops_per_token}."
        )


class TestStrictCausality:
    @given(attention_shapes(), batch_time_shapes)
    @hypothesis_settings
    def test_absolute_temporal_isolation(self, attn_shape, bt_shape):
        d_model, n_heads = attn_shape
        B, T = bt_shape
        assume(T > 1)

        block = make_block(d_model=d_model, n_heads=n_heads, dropout=0.0)
        block.eval()

        x_clean = torch.randn(B, T, d_model)
        out_clean = run_block(block, x_clean, is_causal=True)

        x_corrupted = x_clean.clone()
        split = T // 2
        x_corrupted[:, split:, :] = torch.randn(B, T - split, d_model) * 1000.0

        out_corrupted = run_block(block, x_corrupted, is_causal=True)

        torch.testing.assert_close(
            out_clean[:, :split, :],
            out_corrupted[:, :split, :],
            rtol=0.0,
            atol=1e-5,
            msg="Future tokens influenced earlier positions; the block is not causal.",
        )


class TestBatchEquivariance:
    @given(
        attention_shapes(),
        st.integers(min_value=2, max_value=32),
        st.integers(min_value=1, max_value=128),
    )
    @hypothesis_settings
    def test_batch_permutation_independence(self, attn_shape, B, T):
        d_model, n_heads = attn_shape

        block = make_block(d_model=d_model, n_heads=n_heads, dropout=0.0)
        block.eval()

        x = torch.randn(B, T, d_model)

        perm = torch.randperm(B)
        inv_perm = torch.argsort(perm)

        out_original = run_block(block, x)

        x_shuffled = x[perm]
        out_shuffled = run_block(block, x_shuffled)

        out_restored = out_shuffled[inv_perm]

        torch.testing.assert_close(
            out_original,
            out_restored,
            msg="Output depends on batch ordering; samples are not independent.",
        )


class TestResidualHighway:
    @given(attention_shapes(), batch_time_shapes)
    @hypothesis_settings
    def test_zero_weight_identity_mapping(self, attn_shape, bt_shape):
        d_model, n_heads = attn_shape
        B, T = bt_shape

        block = make_block(d_model=d_model, n_heads=n_heads)

        with torch.no_grad():
            block.out_proj.weight.zero_()
            block.mlp.down_proj.weight.zero_()

        x = torch.randn(B, T, d_model)
        out = run_block(block, x, is_causal=True)

        torch.testing.assert_close(
            out,
            x,
            msg=(
                "Zeroing both output projections did not reduce the block "
                "to the identity."
            ),
        )


class TestAttentionDropout:
    def test_dropout_disabled_in_eval(self):
        block = make_block(d_model=64, n_heads=8, dropout=0.5)
        block.eval()

        x = torch.randn(2, 8, 64)

        with torch.no_grad():
            torch.testing.assert_close(
                run_block(block, x),
                run_block(block, x),
                msg="Eval mode is non-deterministic; dropout leaked into inference.",
            )

    def test_dropout_active_in_train(self):
        block = make_block(d_model=64, n_heads=8, dropout=0.5)
        block.train()

        x = torch.randn(2, 8, 64)

        with torch.no_grad():
            assert not torch.allclose(run_block(block, x), run_block(block, x)), (
                "Train mode is deterministic; dropout is not being applied."
            )

    def test_attention_dropout_reaches_sdpa(self):
        block = make_block(d_model=64, n_heads=8, dropout=0.5)
        block.train()
        block.drop = None

        x = torch.randn(2, 8, 64)

        with torch.no_grad():
            assert not torch.allclose(run_block(block, x), run_block(block, x)), (
                "With residual dropout removed the block is deterministic; "
                "SDPA never saw dropout_p."
            )

    def test_zero_dropout_is_deterministic_in_train(self):
        block = make_block(d_model=64, n_heads=8, dropout=0.0)
        block.train()

        x = torch.randn(2, 8, 64)

        with torch.no_grad():
            torch.testing.assert_close(
                run_block(block, x),
                run_block(block, x),
                msg="Block with dropout=0.0 is non-deterministic in train mode.",
            )


def test_sdpa_causal_mask_collision():
    block = make_block(d_model=32, n_heads=4)
    x = torch.randn(1, 4, 32)
    dummy_mask = torch.ones(4, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="fold causality into mask"):
        run_block(block, x, mask=dummy_mask, is_causal=True)


def test_indivisible_heads():
    with pytest.raises(ValueError, match="not divisible by n_heads"):
        make_block(d_model=100, n_heads=3)


def test_rope_width_must_match_attention_head_width():
    rope = RotaryEmbedding(4, 4096, 10_000.0)
    with pytest.raises(
        ValueError,
        match="rope_head_dim=4 does not match head_dim=3",
    ):
        make_block(d_model=12, n_heads=4, rope=rope)


def test_uncached_mask_must_cover_all_keys():
    block = make_block(d_model=32, n_heads=4)
    x = torch.randn(1, 4, 32)
    short_mask = torch.ones(1, 1, 4, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="mask_key_length=3.*key_length=4"):
        run_block(block, x, mask=short_mask, is_causal=False)
