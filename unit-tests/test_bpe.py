import pytest
import torch

from models.bpe.bpe import BPEModel


def build_model(**overrides):
    config = dict(
        initialiser_range=0.02,
        tie_word_embeddings=True,
    )
    config.update(overrides)
    return BPEModel(
        vocab_size=31,
        d_model=16,
        n_layers=2,
        n_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        multiple_of=8,
        max_seq_len=32,
        rope_theta=10_000.0,
        norm_eps=1e-6,
        **config,
    )


def test_bpe_constructor_ties_input_and_output_embeddings_by_default():
    model = build_model()
    assert model.lm_head.weight is model.embedding.weight
    assert model.embedding.weight.shape == model.lm_head.weight.shape == (31, 16)
    assert isinstance(model.trunk.in_proj, torch.nn.Identity)
    assert isinstance(model.trunk.out, torch.nn.Identity)


def test_bpe_constructor_can_leave_embeddings_untied():
    model = build_model(tie_word_embeddings=False)
    assert model.lm_head.weight is not model.embedding.weight


def test_bpe_uses_configured_embedding_initialisation_scale():
    torch.manual_seed(7)
    model = build_model(initialiser_range=0.01)
    assert abs(model.embedding.weight.std().item() - 0.01) < 0.002


def test_bpe_forward_returns_one_distribution_per_position():
    model = build_model()
    token_ids = torch.randint(0, 31, (3, 7))
    assert model(token_ids).shape == (3, 7, 31)


def test_bpe_loss_matches_cross_entropy_over_next_tokens():
    model = build_model()
    token_ids = torch.randint(0, 31, (3, 7))
    expected = torch.nn.functional.cross_entropy(
        model(token_ids).flatten(0, 1), token_ids.roll(-1, 1).flatten()
    )
    actual = model.loss(token_ids, token_ids.roll(-1, 1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("weight", [1.0, 0.25])
def test_bpe_chunked_backward_matches_materialised_gradients(padding, weight):
    torch.manual_seed(12)
    expected_model = build_model()
    actual_model = build_model()
    actual_model.load_state_dict(expected_model.state_dict())
    token_ids = torch.randint(0, 31, (2, 5))
    targets = token_ids.roll(-1, 1)
    if padding:
        targets[0, :3] = -100
        targets[1, -1] = -100

    expected = expected_model.loss(token_ids, targets)
    (expected * weight).backward()
    states = actual_model.state(token_ids)
    actual = actual_model.backward_chunked_loss(
        states, targets, chunk_rows=3, weight=weight
    )

    torch.testing.assert_close(actual, expected.detach())
    for actual_parameter, expected_parameter in zip(
        actual_model.parameters(), expected_model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


def test_chunked_all_ignored_batch_has_zero_loss_and_gradients():
    model = build_model()
    ids = torch.randint(0, 31, (2, 5))
    loss = model.backward_chunked_loss(
        model.state(ids), torch.full_like(ids, -100), 3, 1.0
    )
    assert loss.item() == 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.grad.count_nonzero().item() == 0
