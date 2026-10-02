import pytest
import torch
from hypothesis import given, strategies as st
from strategies import attention_shapes, hypothesis_settings

from models.shared.block import Block
from models.shared.kv_cache import KVCache
from models.shared.rope import RotaryEmbedding


class TestKVCache:
    @given(attention_shapes(), st.integers(min_value=2, max_value=12))
    @hypothesis_settings
    def test_incremental_decode_matches_full_forward(self, attn_shape, T):
        d_model, n_heads = attn_shape

        block = make_block(d_model=d_model, n_heads=n_heads)
        block.eval()

        x = torch.randn(1, T, d_model)

        with torch.no_grad():
            full = run_block(block, x, is_causal=True)

            cache = KVCache(max_seq_len=1024)
            steps = [run_block(block, x[:, :1], is_causal=True, cache=cache)]
            for t in range(1, T):
                steps.append(
                    run_block(block, x[:, t : t + 1], is_causal=True, cache=cache)
                )
            incremental = torch.cat(steps, dim=1)

        torch.testing.assert_close(
            incremental,
            full,
            rtol=1e-4,
            atol=1e-4,
            msg="Cached decode diverged from the full forward pass.",
        )

    @given(
        attention_shapes(),
        st.integers(min_value=2, max_value=8),
        st.integers(min_value=1, max_value=6),
    )
    @hypothesis_settings
    def test_prefill_then_decode_matches_full_forward(self, attn_shape, prefill, extra):
        d_model, n_heads = attn_shape

        block = make_block(d_model=d_model, n_heads=n_heads)
        block.eval()

        T = prefill + extra
        x = torch.randn(1, T, d_model)

        with torch.no_grad():
            full = run_block(block, x, is_causal=True)

            cache = KVCache(max_seq_len=1024)
            head = run_block(block, x[:, :prefill], is_causal=True, cache=cache)
            tail = [
                run_block(block, x[:, t : t + 1], is_causal=True, cache=cache)
                for t in range(prefill, T)
            ]
            combined = torch.cat([head] + tail, dim=1)

        torch.testing.assert_close(
            combined,
            full,
            rtol=1e-4,
            atol=1e-4,
            msg="Prefill followed by cached decode diverged from the full forward pass.",
        )

    def test_cache_tracks_position(self):
        block = make_block(d_model=64, n_heads=8)
        block.eval()

        cache = KVCache(max_seq_len=1024)
        assert cache.pos == 0

        with torch.no_grad():
            run_block(block, torch.randn(1, 5, 64), cache=cache)
            assert cache.pos == 5

            run_block(block, torch.randn(1, 1, 64), cache=cache)
            assert cache.pos == 6

        cache.reset()
        assert cache.pos == 0

    def test_chunked_prefill_rejected(self):
        block = make_block(d_model=64, n_heads=8)
        block.eval()

        cache = KVCache(max_seq_len=1024)
        with torch.no_grad():
            run_block(block, torch.randn(1, 4, 64), cache=cache)

            with pytest.raises(ValueError, match="one token, not a chunk"):
                run_block(block, torch.randn(1, 3, 64), cache=cache)

    def test_cache_respects_rope_offset(self):
        block = make_block(d_model=64, n_heads=8)
        block.eval()

        x = torch.randn(1, 6, 64)

        with torch.no_grad():
            full = run_block(block, x, is_causal=True)

            cache = KVCache(max_seq_len=1024)
            run_block(block, x[:, :5], is_causal=True, cache=cache)
            stepped = run_block(block, x[:, 5:6], is_causal=True, cache=cache)

        assert not torch.allclose(stepped[:, 0], full[:, 0], atol=1e-3), (
            "Decode step reproduced position 0, indicating the rotary offset was ignored."
        )

        torch.testing.assert_close(
            stepped[:, 0],
            full[:, 5],
            rtol=1e-4,
            atol=1e-4,
            msg="Decode step did not reproduce its true position under rotary embedding.",
        )


class TestKVCacheAllocation:
    @pytest.mark.parametrize("max_seq_len", [0, -1])
    def test_rejects_non_positive_capacity(self, max_seq_len):
        with pytest.raises(ValueError, match=rf"max_seq_len={max_seq_len}"):
            KVCache(max_seq_len=max_seq_len)

    def test_rejects_non_integer_capacity(self):
        with pytest.raises(TypeError, match="max_seq_len_type=float"):
            KVCache(max_seq_len=8.0)

    def test_rejects_zero_length_update(self):
        cache = KVCache(max_seq_len=8)

        with pytest.raises(ValueError, match="T=0"):
            cache.update(torch.randn(1, 2, 0, 8), torch.randn(1, 2, 0, 8))

    def test_remaining_tracks_reset_and_free(self):
        cache = KVCache(max_seq_len=8)
        assert cache.remaining == 8

        cache.update(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
        assert cache.remaining == 5

        cache.reset()
        assert cache.remaining == 8

        cache.update(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8))
        cache.free()
        assert cache.remaining == 8

    def test_buffer_allocated_once(self):
        block = make_block(d_model=64, n_heads=8)
        block.eval()

        cache = KVCache(max_seq_len=32)
        with torch.no_grad():
            run_block(block, torch.randn(1, 4, 64), cache=cache)
            first = cache.k.data_ptr()

            for t in range(6):
                run_block(block, torch.randn(1, 1, 64), cache=cache)

        assert cache.k.data_ptr() == first, "Cache reallocated during decode."
        assert cache.k.shape[-2] == 32, "Cache buffer is not the preallocated length."

    def test_overflow_raises(self):
        block = make_block(d_model=64, n_heads=8)
        block.eval()

        cache = KVCache(max_seq_len=4)
        with torch.no_grad():
            run_block(block, torch.randn(1, 4, 64), cache=cache)

            with pytest.raises(ValueError, match="max_seq_len=4, required_positions=5"):
                run_block(block, torch.randn(1, 1, 64), cache=cache)

    def test_reset_reuses_buffer(self):
        block = make_block(d_model=64, n_heads=8)
        block.eval()

        cache = KVCache(max_seq_len=16)
        with torch.no_grad():
            run_block(block, torch.randn(1, 5, 64), cache=cache)
            ptr = cache.k.data_ptr()

            cache.reset()
            assert cache.pos == 0

            run_block(block, torch.randn(1, 5, 64), cache=cache)

        assert cache.k.data_ptr() == ptr, "reset() dropped the preallocated buffer."

    def test_reset_rejects_a_different_batch_shape(self):
        cache = KVCache(max_seq_len=16)
        cache.update(torch.randn(4, 2, 3, 8), torch.randn(4, 2, 3, 8))
        cache.reset()

        with pytest.raises(ValueError, match="cache_shape=.*input_shape=.*free"):
            cache.update(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))

    def test_free_allows_shape_change_and_releases_storage(self):
        cache = KVCache(max_seq_len=16)
        cache.update(torch.randn(4, 2, 3, 8), torch.randn(4, 2, 3, 8))

        cache.free()

        assert cache.k is None
        assert cache.v is None
        assert cache.pos == 0
        keys, values = cache.update(
            torch.randn(1, 3, 2, 4),
            torch.randn(1, 3, 2, 4),
        )
        assert keys.shape == values.shape == (1, 3, 2, 4)

    def test_update_does_not_build_an_autograd_graph(self):
        cache = KVCache(max_seq_len=16)
        k = torch.randn(1, 2, 3, 8, requires_grad=True)
        v = torch.randn(1, 2, 3, 8, requires_grad=True)

        keys, values = cache.update(k, v)

        assert not keys.requires_grad
        assert not values.requires_grad
        assert keys.grad_fn is None
        assert values.grad_fn is None

    def test_returned_views_alias_cache_storage(self):
        cache = KVCache(max_seq_len=16)
        keys, values = cache.update(
            torch.randn(1, 2, 3, 8),
            torch.randn(1, 2, 3, 8),
        )

        assert keys.untyped_storage().data_ptr() == cache.k.untyped_storage().data_ptr()
        assert (
            values.untyped_storage().data_ptr() == cache.v.untyped_storage().data_ptr()
        )

    def test_rejects_dtype_change_after_allocation(self):
        cache = KVCache(max_seq_len=16)
        cache.update(
            torch.randn(1, 2, 3, 8, dtype=torch.float32),
            torch.randn(1, 2, 3, 8, dtype=torch.float32),
        )

        with pytest.raises(ValueError, match="cache_dtype=torch.float32"):
            cache.update(
                torch.randn(1, 2, 1, 8, dtype=torch.float64),
                torch.randn(1, 2, 1, 8, dtype=torch.float64),
            )

    @pytest.mark.parametrize(
        ("k", "v", "message"),
        [
            (torch.randn(2, 3, 4), torch.randn(2, 3, 4), "expected k shape"),
            (
                torch.randn(1, 2, 3, 4),
                torch.randn(1, 2, 2, 4),
                "k_shape=.*v_shape",
            ),
            (
                torch.randn(1, 2, 3, 4, dtype=torch.float32),
                torch.randn(1, 2, 3, 4, dtype=torch.float64),
                "k_dtype=.*v_dtype",
            ),
        ],
    )
    def test_rejects_incompatible_inputs(self, k, v, message):
        cache = KVCache(max_seq_len=16)

        with pytest.raises(ValueError, match=message):
            cache.update(k, v)

    def test_reset_does_not_leak_stale_keys(self):
        block = make_block(d_model=64, n_heads=8)
        block.eval()

        x = torch.randn(1, 6, 64)

        with torch.no_grad():
            fresh = KVCache(max_seq_len=16)
            clean = run_block(block, x, is_causal=True, cache=fresh)

            reused = KVCache(max_seq_len=16)
            run_block(block, torch.randn(1, 9, 64), is_causal=True, cache=reused)
            reused.reset()
            after = run_block(block, x, is_causal=True, cache=reused)

        torch.testing.assert_close(
            after,
            clean,
            rtol=1e-4,
            atol=1e-4,
            msg="Reset cache still attended to keys from the previous sequence.",
        )

    def test_cache_matches_dtype_of_keys(self):
        block = make_block(d_model=64, n_heads=8).to(torch.float64)
        block.eval()

        cache = KVCache(max_seq_len=8)
        with torch.no_grad():
            run_block(block, torch.randn(1, 3, 64, dtype=torch.float64), cache=cache)

        assert cache.k.dtype == torch.float64, (
            "Cache buffer dtype does not follow the model."
        )


def test_cached_mask_must_cover_all_cached_keys():
    block = make_block(d_model=64, n_heads=8)
    block.eval()
    cache = KVCache(max_seq_len=8)

    with torch.no_grad():
        run_block(block, torch.randn(1, 3, 64), cache=cache)
        stale_mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
        with pytest.raises(
            ValueError,
            match="mask_key_length=3 does not match key_length=4",
        ):
            run_block(
                block,
                torch.randn(1, 1, 64),
                mask=stale_mask,
                is_causal=False,
                cache=cache,
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
