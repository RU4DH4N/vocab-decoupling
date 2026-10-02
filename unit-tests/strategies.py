from hypothesis import settings, strategies as st


@st.composite
def attention_shapes(draw):
    head_dim = draw(st.sampled_from([8, 16, 32, 64]))
    n_heads = draw(st.integers(min_value=1, max_value=8))

    return head_dim * n_heads, n_heads


swiglu_config = st.tuples(
    st.integers(min_value=16, max_value=1024),
    st.floats(min_value=0.1, max_value=10.0),
    st.sampled_from([8, 16, 32, 64, 128, 256]),
)

batch_time_shapes = st.tuples(
    st.integers(min_value=1, max_value=8), st.integers(min_value=1, max_value=64)
)

hypothesis_settings = settings(max_examples=50, deadline=None)
