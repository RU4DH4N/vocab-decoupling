import pytest
import torch
from hypothesis import given, strategies as st
from strategies import attention_shapes, batch_time_shapes, hypothesis_settings

from models.shared.block import Block
from models.shared.rope import RotaryEmbedding, rotate_half


class TestRotaryEmbedding:
    @given(st.sampled_from([8, 16, 32, 64]), st.integers(min_value=4, max_value=32))
    @hypothesis_settings
    def test_relative_position_dependence(self, head_dim, T):
        rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=T,
            theta=10_000.0,
        )

        qv = torch.randn(head_dim)
        kv = torch.randn(head_dim)
        q = qv.view(1, 1, 1, head_dim).expand(1, 1, T, head_dim).contiguous()
        k = kv.view(1, 1, 1, head_dim).expand(1, 1, T, head_dim).contiguous()

        qr, kr = rope(q, k, offset=0)

        def score(m, n):
            return (qr[0, 0, m] * kr[0, 0, n]).sum()

        for m, n, shift in ((0, 1, 2), (1, 3, 1), (0, 0, T - 1)):
            if max(m, n) + shift >= T:
                continue
            torch.testing.assert_close(
                score(m, n),
                score(m + shift, n + shift),
                rtol=0.0,
                atol=1e-4,
                msg="Rotary attention score depends on absolute rather than relative offset.",
            )

    @given(attention_shapes(), batch_time_shapes)
    @hypothesis_settings
    def test_rotation_preserves_norm(self, attn_shape, bt_shape):
        d_model, n_heads = attn_shape
        B, T = bt_shape
        head_dim = d_model // n_heads

        rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=max(T, 1),
            theta=10_000.0,
        )

        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)

        qr, kr = rope(q, k, offset=0)

        torch.testing.assert_close(
            qr.norm(dim=-1),
            q.norm(dim=-1),
            rtol=1e-4,
            atol=1e-4,
            msg="Rotary embedding is not norm preserving.",
        )
        torch.testing.assert_close(
            kr.norm(dim=-1),
            k.norm(dim=-1),
            rtol=1e-4,
            atol=1e-4,
            msg="Rotary embedding is not norm preserving.",
        )

    @given(st.sampled_from([8, 16, 32, 64]))
    @hypothesis_settings
    def test_positions_are_distinguished(self, head_dim):
        rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=16,
            theta=10_000.0,
        )

        qv = torch.randn(head_dim)
        q = qv.view(1, 1, 1, head_dim).expand(1, 1, 16, head_dim).contiguous()

        qr, _ = rope(q, q, offset=0)

        assert not torch.allclose(qr[0, 0, 0], qr[0, 0, 1], atol=1e-6), (
            "Rotary embedding leaves distinct positions identical."
        )

    @given(st.sampled_from([8, 16, 32]))
    @hypothesis_settings
    def test_lazy_extension_beyond_cache(self, head_dim):
        rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=8,
            theta=10_000.0,
        )

        q = torch.randn(1, 2, 40, head_dim)
        qr, kr = rope(q, q, offset=0)

        assert rope.cos.shape[0] >= 40, (
            "Rotary cache failed to extend past max_seq_len."
        )
        assert qr.shape == q.shape

    def test_cache_growth_is_geometric_and_remains_registered(self):
        rope = RotaryEmbedding(
            head_dim=8,
            max_seq_len=8,
            theta=10_000.0,
        )
        q = torch.randn(1, 2, 9, 8)

        rope(q, q, offset=0)

        assert rope.configured_max_seq_len == 8
        assert rope.cos.shape[0] == 16
        assert rope._buffers["cos"] is rope.cos
        assert rope._buffers["sin"] is rope.sin

    def test_negative_offset_is_rejected(self):
        rope = RotaryEmbedding(
            head_dim=8,
            max_seq_len=8,
            theta=10_000.0,
        )
        q = torch.randn(1, 2, 1, 8)

        with pytest.raises(ValueError, match="offset=-1 must be non-negative"):
            rope(q, q, offset=-1)

    @pytest.mark.parametrize(
        ("q_shape", "k_shape", "message"),
        [
            ((1, 2, 4, 6), (1, 2, 4, 8), "q_head_dim=6.*head_dim=8"),
            ((1, 2, 4, 8), (1, 2, 4, 6), "k_head_dim=6.*head_dim=8"),
            (
                (1, 2, 3, 8),
                (1, 2, 4, 8),
                "q_sequence_length=3.*k_sequence_length=4",
            ),
        ],
    )
    def test_rejects_incompatible_tensor_shapes(
        self,
        q_shape,
        k_shape,
        message,
    ):
        rope = RotaryEmbedding(
            head_dim=8,
            max_seq_len=8,
            theta=10_000.0,
        )

        with pytest.raises(ValueError, match=message):
            rope(torch.randn(q_shape), torch.randn(k_shape), offset=0)

    def test_odd_head_dim_rejected(self):
        with pytest.raises(ValueError, match="must be even"):
            RotaryEmbedding(
                head_dim=33,
                max_seq_len=4096,
                theta=10_000.0,
            )

    def test_block_surfaces_odd_head_dim(self):
        with pytest.raises(ValueError, match="must be even"):
            make_block(d_model=18, n_heads=2)

    def test_rope_contributes_no_parameters(self):
        rope = RotaryEmbedding(
            head_dim=64,
            max_seq_len=4096,
            theta=10_000.0,
        )

        assert list(rope.parameters()) == [], (
            "Rotary embedding introduced parameters, invalidating the FLOP accounting."
        )

    def test_rope_buffers_are_not_persistent(self):
        keys = make_block(d_model=64, n_heads=8).state_dict().keys()

        assert not [k for k in keys if "rope" in k], (
            "Rotary caches leaked into state_dict; checkpoints break on sequence length change."
        )

    @given(attention_shapes())
    @hypothesis_settings
    def test_shared_rope_matches_private_rope(self, attn_shape):
        d_model, n_heads = attn_shape

        shared = RotaryEmbedding(
            head_dim=d_model // n_heads,
            max_seq_len=32,
            theta=10_000.0,
        )

        private_block = make_block(d_model=d_model, n_heads=n_heads)
        shared_block = make_block(d_model=d_model, n_heads=n_heads, rope=shared)
        shared_block.load_state_dict(private_block.state_dict())

        private_block.eval()
        shared_block.eval()

        x = torch.randn(2, 12, d_model)

        torch.testing.assert_close(
            run_block(private_block, x),
            run_block(shared_block, x),
            msg="Sharing one rotary cache across blocks changed the result.",
        )


class TestRotateHalf:
    @given(st.sampled_from([8, 16, 32, 64]))
    @hypothesis_settings
    def test_rotate_half_is_quarter_turn(self, head_dim):
        x = torch.randn(2, 3, head_dim)

        assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6), (
            "rotate_half applied twice is not negation."
        )


def make_block(d_model, n_heads, **overrides):
    config = {
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "multiple_of": 32,
        "rope": RotaryEmbedding(d_model // n_heads, 4096, 10_000.0),
        "norm_eps": 1e-6,
    }
    config.update(overrides)
    return Block(d_model, n_heads, **config)


def run_block(block, x, *, mask=None, is_causal=True, cache=None):
    return block(x, mask=mask, is_causal=is_causal, cache=cache)
