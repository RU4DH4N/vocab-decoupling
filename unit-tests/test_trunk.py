import pytest
import torch
from hypothesis import given, strategies as st
from strategies import hypothesis_settings

from models.shared.block import Block
from models.shared.trunk import Trunk, TrunkConfig


@st.composite
def trunk_shapes(draw):
    head_dim = draw(st.sampled_from([8, 16, 32]))
    n_heads = draw(st.integers(min_value=1, max_value=8))
    n_layers = draw(st.integers(min_value=1, max_value=4))
    return head_dim * n_heads, n_heads, n_layers


def build(d_model, n_heads, n_layers, in_dims=64, out_dims=32, **kw):
    config = {
        "initialiser_range": 0.02,
        "dropout": 0.0,
        "multiple_of": 32,
        "max_seq_len": 4096,
        "rope_theta": 10_000.0,
        "norm_eps": 1e-6,
    }
    config.update(kw)
    return Trunk(
        TrunkConfig(
            in_dims=in_dims,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=4.0,
            out_dims=out_dims,
            **config,
        )
    )


class TestTrunkCausality:
    @given(trunk_shapes(), st.integers(min_value=2, max_value=32))
    @hypothesis_settings
    def test_future_does_not_leak_into_prefix(self, shape, T):
        d_model, n_heads, n_layers = shape

        trunk = build(d_model, n_heads, n_layers)
        trunk.eval()

        x = torch.randn(2, T, 64)
        split = T // 2

        corrupted = x.clone()
        corrupted[:, split:, :] = torch.randn(2, T - split, 64) * 1000.0

        with torch.no_grad():
            clean = trunk(x)
            dirty = trunk(corrupted)

        torch.testing.assert_close(
            clean[:, :split, :],
            dirty[:, :split, :],
            rtol=0.0,
            atol=1e-4,
            msg="Future positions influenced the prefix: the trunk is not causal.",
        )

    def test_every_position_depends_only_on_its_past(self):
        trunk = build(64, 8, 3)
        trunk.eval()

        T = 8
        x = torch.randn(1, T, 64)

        with torch.no_grad():
            base = trunk(x)

            for t in range(T):
                bumped = x.clone()
                bumped[:, t, :] += 50.0
                out = trunk(bumped)

                if t > 0:
                    torch.testing.assert_close(
                        out[:, :t, :],
                        base[:, :t, :],
                        rtol=0.0,
                        atol=1e-4,
                        msg=f"Editing position {t} changed an earlier position.",
                    )
                assert not torch.allclose(out[:, t, :], base[:, t, :], atol=1e-5), (
                    f"Editing position {t} did not change position {t} itself."
                )


class TestRopeSharing:
    @given(trunk_shapes())
    @hypothesis_settings
    def test_one_rope_instance_across_all_blocks(self, shape):
        d_model, n_heads, n_layers = shape

        trunk = build(d_model, n_heads, n_layers)

        assert len({id(b.rope) for b in trunk.blocks}) == 1, (
            "Blocks hold separate rotary caches; the shared instance was not propagated."
        )

    def test_rope_absent_from_state_dict(self):
        keys = build(64, 8, 4).state_dict().keys()

        assert not [k for k in keys if "rope" in k], (
            "Rotary caches leaked into state_dict; checkpoints break on length change."
        )


class TestInitialisation:
    def test_residual_output_scale_accounts_for_depth(self):
        torch.manual_seed(7)
        trunk = build(16, 2, 2, initialiser_range=0.02)
        expected = 0.02 / (2 * 2) ** 0.5
        assert abs(trunk.blocks[0].out_proj.weight.std().item() - expected) < 0.003
        assert abs(trunk.blocks[0].mlp.down_proj.weight.std().item() - expected) < 0.003

    def test_residual_outputs_identify_their_own_initialisation_role(self):
        trunk = build(16, 2, 2)
        tagged = [
            module
            for module in trunk.modules()
            if getattr(module, "residual_output", False)
        ]
        assert tagged == [
            projection
            for block in trunk.blocks
            for projection in (block.out_proj, block.mlp.down_proj)
        ]

    def test_missing_residual_marker_fails_construction(self, monkeypatch):

        original = Block.__init__

        def without_attention_marker(block, *args, **kwargs):
            original(block, *args, **kwargs)
            block.out_proj = torch.nn.Linear(
                block.d_model,
                block.d_model,
                bias=False,
            )

        monkeypatch.setattr(Block, "__init__", without_attention_marker)
        with pytest.raises(
            RuntimeError, match="expected 4 residual-output projections, n_flagged=2"
        ):
            build(16, 2, 2)


class TestSeamProjections:
    def test_equal_dimensions_use_identity_seams(self):
        trunk = build(16, 2, 2, in_dims=16, out_dims=16)
        assert isinstance(trunk.in_proj, torch.nn.Identity)
        assert isinstance(trunk.out, torch.nn.Identity)

    def test_dimension_changes_use_bias_free_projections(self):
        trunk = build(16, 2, 2, in_dims=8, out_dims=4)
        assert isinstance(trunk.in_proj, torch.nn.Linear)
        assert isinstance(trunk.out, torch.nn.Linear)
        assert trunk.in_proj.bias is None
        assert trunk.out.bias is None


class TestConfigPropagation:
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("d_model", 0, "d_model must be positive"),
            ("mlp_ratio", 0.0, "mlp_ratio must be positive"),
            ("dropout", -0.1, "dropout must be in"),
            ("dropout", 1.0, "dropout must be in"),
            ("dropout", 1.1, "dropout must be in"),
        ],
    )
    def test_invalid_scalar_config_is_rejected(self, field, value, message):
        config = {
            "in_dims": 16,
            "d_model": 16,
            "n_layers": 2,
            "n_heads": 2,
            "mlp_ratio": 4.0,
            "out_dims": 16,
            "initialiser_range": 0.02,
            "dropout": 0.0,
            "multiple_of": 8,
            "max_seq_len": 32,
            "rope_theta": 10_000.0,
            "norm_eps": 1e-6,
        }
        config[field] = value
        with pytest.raises(ValueError, match=message):
            TrunkConfig(**config)

    @given(st.floats(min_value=0.05, max_value=0.9))
    @hypothesis_settings
    def test_dropout_reaches_every_block(self, p):
        trunk = build(64, 8, 4, dropout=p)

        assert all(b.p_drop == pytest.approx(p) for b in trunk.blocks)
        assert all(b.drop is not None for b in trunk.blocks)

    @given(st.sampled_from([8, 16, 32, 64]))
    @hypothesis_settings
    def test_mlp_width_reaches_every_block(self, multiple_of):
        trunk = build(64, 8, 3, multiple_of=multiple_of)

        assert all(b.mlp.hidden_dim % multiple_of == 0 for b in trunk.blocks)
        assert len({b.mlp.hidden_dim for b in trunk.blocks}) == 1

    def test_indivisible_heads_reports_the_useful_error(self):
        with pytest.raises(ValueError, match="not divisible by n_heads"):
            build(100, 3, 1)

    def test_nonpositive_head_count_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="n_heads must be positive"):
            build(64, 0, 1)

    def test_odd_rope_head_dimension_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="RoPE requires an even head dimension"):
            build(30, 2, 1)


class TestSequenceLength:
    def test_configured_positional_limit_is_enforced(self):
        trunk = build(64, 8, 2, max_seq_len=32)
        with pytest.raises(ValueError, match="sequence length 33.*maximum 32"):
            trunk(torch.randn(1, 33, 64))

    def test_configured_positional_limit_is_inclusive(self):
        trunk = build(64, 8, 2, max_seq_len=16)
        assert trunk(torch.randn(1, 16, 64)).shape == (1, 16, 32)


class TestTrunkShape:
    @given(
        trunk_shapes(),
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=24),
    )
    @hypothesis_settings
    def test_projects_in_dims_to_out_dims(self, shape, B, T):
        d_model, n_heads, n_layers = shape

        trunk = build(d_model, n_heads, n_layers, in_dims=64, out_dims=32)
        trunk.eval()

        with torch.no_grad():
            out = trunk(torch.randn(B, T, 64))

        assert out.shape == (B, T, 32)

    @given(trunk_shapes(), st.integers(min_value=2, max_value=8))
    @hypothesis_settings
    def test_batch_permutation_independence(self, shape, B):
        d_model, n_heads, n_layers = shape

        trunk = build(d_model, n_heads, n_layers)
        trunk.eval()

        x = torch.randn(B, 10, 64)
        perm = torch.randperm(B)

        with torch.no_grad():
            straight = trunk(x)[perm]
            shuffled = trunk(x[perm])

        torch.testing.assert_close(
            straight,
            shuffled,
            msg="Output depends on batch ordering; samples are not independent.",
        )
