import pytest
import torch
from hypothesis import given, settings, strategies as st
from strategies import hypothesis_settings, swiglu_config
from torch.autograd import gradcheck

from models.shared.swiglu import SwiGLU, swiglu_hidden_dim


class TestSwiGLUShape:
    @given(swiglu_config)
    @hypothesis_settings
    def test_swiglu_hidden_dim_alignment(self, config):
        d_model, mlp_ratio, multiple_of = config

        swiglu = SwiGLU(d_model=d_model, mlp_ratio=mlp_ratio, multiple_of=multiple_of)

        assert swiglu.hidden_dim % multiple_of == 0
        assert swiglu.gate_proj.weight.shape == (swiglu.hidden_dim, d_model)
        assert swiglu.up_proj.weight.shape == (swiglu.hidden_dim, d_model)
        assert swiglu.down_proj.weight.shape == (d_model, swiglu.hidden_dim)
        assert swiglu.gate_proj.bias is None
        assert swiglu.up_proj.bias is None
        assert swiglu.down_proj.bias is None


class TestAnalyticalGradients:
    @given(
        st.integers(min_value=8, max_value=32),
        st.integers(min_value=1, max_value=4),
        st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=20, deadline=None)
    def test_swiglu_jacobian_gradcheck(self, d_model, B, T):
        swiglu = SwiGLU(
            d_model=d_model,
            mlp_ratio=4.0,
            multiple_of=8,
        ).to(torch.float64)
        x = torch.randn(B, T, d_model, dtype=torch.float64, requires_grad=True)

        gradcheck(swiglu, (x,), eps=1e-6, atol=1e-4)


def test_hidden_dim_helper_matches_module():
    module = SwiGLU(d_model=96, mlp_ratio=3.5, multiple_of=32)

    assert module.hidden_dim == swiglu_hidden_dim(96, 3.5, 32)


def test_hidden_dim_rounds_up_instead_of_to_nearest():
    assert swiglu_hidden_dim(1024, 4.0, 512) == 3072


@pytest.mark.parametrize(
    ("d_model", "mlp_ratio", "multiple_of", "name"),
    [
        (0, 4.0, 32, "d_model"),
        (64, 0.0, 32, "mlp_ratio"),
        (64, 4.0, 0, "multiple_of"),
    ],
)
def test_hidden_dim_rejects_non_positive_inputs(
    d_model: int,
    mlp_ratio: float,
    multiple_of: int,
    name: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be positive"):
        swiglu_hidden_dim(d_model, mlp_ratio, multiple_of)


@given(
    st.integers(min_value=1, max_value=2048),
    st.integers(min_value=1, max_value=2048),
    st.floats(min_value=0.25, max_value=8.0, allow_nan=False),
    st.sampled_from([8, 16, 32, 64, 128]),
)
@hypothesis_settings
def test_hidden_dim_is_non_decreasing_in_model_width(
    first: int,
    second: int,
    mlp_ratio: float,
    multiple_of: int,
) -> None:
    lower, upper = sorted((first, second))

    assert swiglu_hidden_dim(lower, mlp_ratio, multiple_of) <= swiglu_hidden_dim(
        upper,
        mlp_ratio,
        multiple_of,
    )


def test_constructor_rejects_ambiguous_positional_arguments():
    with pytest.raises(TypeError):
        SwiGLU(96, 3.5, 32)
